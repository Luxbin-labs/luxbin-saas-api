"""
LUXBIN SaaS API - Stripe Payments Integration
==============================================
Handles subscriptions, checkout, and webhooks for the API tiers.

Tiers:
- Free: $0/mo - 100 requests/day
- Pro: $29/mo - 10,000 requests/day + Real Quantum RNG
- Enterprise: $299/mo - Unlimited + Dedicated backend
"""

import os
import stripe
from datetime import datetime
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Your domain for redirects
DOMAIN = os.getenv("DOMAIN", "http://localhost:8000")

router = APIRouter(prefix="/api/v1/billing", tags=["Billing"])

# ============================================================================
# Stripe Product/Price IDs (create these in Stripe Dashboard)
# ============================================================================

STRIPE_PRICES = {
    "pro_monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY", "price_pro_monthly"),
    "pro_yearly": os.getenv("STRIPE_PRICE_PRO_YEARLY", "price_pro_yearly"),
    "enterprise_monthly": os.getenv("STRIPE_PRICE_ENTERPRISE_MONTHLY", "price_enterprise_monthly"),
    "enterprise_yearly": os.getenv("STRIPE_PRICE_ENTERPRISE_YEARLY", "price_enterprise_yearly"),
}

# In-memory subscription store (use database in production)
SUBSCRIPTIONS = {}
CUSTOMERS = {}

# ============================================================================
# Models
# ============================================================================

class CreateCheckoutRequest(BaseModel):
    plan: str = Field(..., description="Plan: pro_monthly, pro_yearly, enterprise_monthly, enterprise_yearly")
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class CreateCheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str

class SubscriptionStatus(BaseModel):
    api_key: str
    plan: str
    status: str
    current_period_end: Optional[str]
    cancel_at_period_end: bool

class CreatePortalRequest(BaseModel):
    return_url: Optional[str] = None

# ============================================================================
# Endpoints
# ============================================================================

@router.post("/checkout", response_model=CreateCheckoutResponse)
async def create_checkout_session(
    request: CreateCheckoutRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Create a Stripe Checkout session for upgrading to a paid plan.

    Returns a URL to redirect the user to Stripe's hosted checkout page.
    """
    if not stripe.api_key:
        raise HTTPException(503, "Stripe not configured")

    if request.plan not in STRIPE_PRICES:
        raise HTTPException(400, f"Invalid plan. Choose from: {list(STRIPE_PRICES.keys())}")

    price_id = STRIPE_PRICES[request.plan]

    try:
        # Create or get customer
        customer_id = CUSTOMERS.get(x_api_key)
        if not customer_id:
            customer = stripe.Customer.create(
                metadata={"api_key": x_api_key}
            )
            customer_id = customer.id
            CUSTOMERS[x_api_key] = customer_id

        # Create checkout session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="subscription",
            success_url=request.success_url or f"{DOMAIN}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=request.cancel_url or f"{DOMAIN}/billing/cancel",
            metadata={
                "api_key": x_api_key,
                "plan": request.plan
            },
            subscription_data={
                "metadata": {
                    "api_key": x_api_key,
                    "plan": request.plan
                }
            }
        )

        return CreateCheckoutResponse(
            checkout_url=session.url,
            session_id=session.id
        )

    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e))

@router.get("/subscription")
async def get_subscription_status(
    x_api_key: str = Header(..., alias="X-API-Key")
) -> SubscriptionStatus:
    """
    Get the current subscription status for an API key.
    """
    sub = SUBSCRIPTIONS.get(x_api_key)

    if not sub:
        return SubscriptionStatus(
            api_key=x_api_key[:10] + "...",
            plan="free",
            status="active",
            current_period_end=None,
            cancel_at_period_end=False
        )

    return SubscriptionStatus(
        api_key=x_api_key[:10] + "...",
        plan=sub.get("plan", "free"),
        status=sub.get("status", "active"),
        current_period_end=sub.get("current_period_end"),
        cancel_at_period_end=sub.get("cancel_at_period_end", False)
    )

@router.post("/portal")
async def create_customer_portal(
    request: CreatePortalRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Create a Stripe Customer Portal session for managing subscription.

    Users can update payment methods, cancel, or change plans.
    """
    if not stripe.api_key:
        raise HTTPException(503, "Stripe not configured")

    customer_id = CUSTOMERS.get(x_api_key)
    if not customer_id:
        raise HTTPException(404, "No subscription found for this API key")

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.return_url or f"{DOMAIN}/dashboard"
        )
        return {"portal_url": session.url}
    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e))

@router.post("/cancel")
async def cancel_subscription(
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Cancel subscription at the end of the current billing period.
    """
    if not stripe.api_key:
        raise HTTPException(503, "Stripe not configured")

    sub = SUBSCRIPTIONS.get(x_api_key)
    if not sub or not sub.get("subscription_id"):
        raise HTTPException(404, "No active subscription found")

    try:
        stripe.Subscription.modify(
            sub["subscription_id"],
            cancel_at_period_end=True
        )
        SUBSCRIPTIONS[x_api_key]["cancel_at_period_end"] = True

        return {"message": "Subscription will cancel at period end", "status": "canceling"}
    except stripe.error.StripeError as e:
        raise HTTPException(400, str(e))

# ============================================================================
# Webhook Handler
# ============================================================================

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events for subscription lifecycle.

    Events handled:
    - checkout.session.completed: New subscription created
    - customer.subscription.updated: Plan changed
    - customer.subscription.deleted: Subscription canceled
    - invoice.payment_failed: Payment failed
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            raise HTTPException(400, "Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(400, "Invalid signature")
    else:
        # No webhook secret configured, parse directly (dev mode)
        import json
        event = json.loads(payload)

    event_type = event.get("type") if isinstance(event, dict) else event.type
    data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event.data.object

    # Handle events
    if event_type == "checkout.session.completed":
        # New subscription created
        api_key = data.get("metadata", {}).get("api_key")
        plan = data.get("metadata", {}).get("plan", "pro_monthly")
        subscription_id = data.get("subscription")

        if api_key:
            # Get subscription details
            sub = stripe.Subscription.retrieve(subscription_id)
            SUBSCRIPTIONS[api_key] = {
                "subscription_id": subscription_id,
                "plan": plan.replace("_monthly", "").replace("_yearly", ""),
                "status": sub.status,
                "current_period_end": datetime.fromtimestamp(sub.current_period_end).isoformat(),
                "cancel_at_period_end": sub.cancel_at_period_end
            }
            print(f"✅ New subscription: {api_key} -> {plan}")

    elif event_type == "customer.subscription.updated":
        # Find API key from subscription metadata
        api_key = data.get("metadata", {}).get("api_key")
        if api_key and api_key in SUBSCRIPTIONS:
            SUBSCRIPTIONS[api_key].update({
                "status": data.get("status"),
                "current_period_end": datetime.fromtimestamp(data.get("current_period_end")).isoformat(),
                "cancel_at_period_end": data.get("cancel_at_period_end", False)
            })
            print(f"📝 Subscription updated: {api_key}")

    elif event_type == "customer.subscription.deleted":
        api_key = data.get("metadata", {}).get("api_key")
        if api_key and api_key in SUBSCRIPTIONS:
            SUBSCRIPTIONS[api_key]["status"] = "canceled"
            SUBSCRIPTIONS[api_key]["plan"] = "free"
            print(f"❌ Subscription canceled: {api_key}")

    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer")
        # Find API key by customer ID
        for api_key, cid in CUSTOMERS.items():
            if cid == customer_id and api_key in SUBSCRIPTIONS:
                SUBSCRIPTIONS[api_key]["status"] = "past_due"
                print(f"⚠️ Payment failed: {api_key}")
                break

    return {"received": True}

# ============================================================================
# Pricing Info Endpoint
# ============================================================================

@router.get("/pricing")
async def get_pricing():
    """
    Get current pricing information for all tiers.
    """
    return {
        "tiers": {
            "free": {
                "price": 0,
                "requests_per_day": 100,
                "features": [
                    "Code translation (all languages)",
                    "Light encoding/decoding",
                    "Simulated quantum RNG",
                    "Community support"
                ]
            },
            "pro": {
                "price_monthly": 29,
                "price_yearly": 290,
                "requests_per_day": 10000,
                "features": [
                    "Everything in Free",
                    "Real IBM Quantum RNG",
                    "Priority queue",
                    "Email support",
                    "Webhook notifications"
                ]
            },
            "enterprise": {
                "price_monthly": 299,
                "price_yearly": 2990,
                "requests_per_day": "unlimited",
                "features": [
                    "Everything in Pro",
                    "Dedicated quantum backend",
                    "Custom integrations",
                    "SLA guarantee",
                    "24/7 support"
                ]
            }
        },
        "currency": "USD"
    }

# ============================================================================
# Helper to check subscription tier
# ============================================================================

def get_user_tier(api_key: str) -> str:
    """Get the subscription tier for an API key"""
    sub = SUBSCRIPTIONS.get(api_key)
    if not sub or sub.get("status") not in ["active", "trialing"]:
        return "free"
    return sub.get("plan", "free")
