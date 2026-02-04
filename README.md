<p align="center">
  <img src="https://img.shields.io/badge/🌈_LUXBIN-API-blueviolet?style=for-the-badge" alt="LUXBIN API"/>
  <img src="https://img.shields.io/badge/⚛️_QUANTUM-RNG-00d4aa?style=for-the-badge" alt="Quantum RNG"/>
  <img src="https://img.shields.io/badge/🔄_CODE-TRANSLATION-ff6b35?style=for-the-badge" alt="Code Translation"/>
</p>

<h1 align="center">LUXBIN SaaS API</h1>

<p align="center">
  <b>Quantum-Enhanced Developer Tools for Web3 & Crypto</b><br>
  <i>AI Code Translation • Light Language Encoding • True Quantum Random Numbers</i>
</p>

---

## Features

| Endpoint | Description | Use Case |
|:---------|:------------|:---------|
| `/api/v1/translate` | AI-powered code translation | Convert Python to Solidity, JS to Rust, etc. |
| `/api/v1/encode` | Encode text to LUXBIN Light Language | Visual data encoding, cross-platform communication |
| `/api/v1/decode` | Decode light wavelengths to text | Receive light-encoded messages |
| `/api/v1/quantum/random` | True quantum random numbers | Cryptographic randomness, NFT minting, gaming |

---

## Quick Start

### 1. Install & Run

```bash
# Clone
git clone https://github.com/nichechristie/luxbin-saas-api.git
cd luxbin-saas-api

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python -m uvicorn src.main:app --reload
```

### 2. Get an API Key

```bash
curl -X POST http://localhost:8000/api/v1/keys/generate?tier=free
```

### 3. Make Requests

```bash
# Translate Python to JavaScript
curl -X POST http://localhost:8000/api/v1/translate \
  -H "X-API-Key: lux_demo_free_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello(name):\n    return f\"Hello, {name}!\"",
    "source_language": "python",
    "target_language": "javascript"
  }'

# Encode text to Light Language
curl -X POST http://localhost:8000/api/v1/encode \
  -H "X-API-Key: lux_demo_free_12345" \
  -H "Content-Type: application/json" \
  -d '{"text": "HELLO"}'

# Get quantum random numbers
curl -X POST http://localhost:8000/api/v1/quantum/random \
  -H "X-API-Key: lux_demo_free_12345" \
  -H "Content-Type: application/json" \
  -d '{"count": 5, "min_value": 1, "max_value": 100}'
```

---

## API Reference

### Code Translation

**POST** `/api/v1/translate`

```json
{
  "code": "def add(a, b): return a + b",
  "source_language": "python",
  "target_language": "rust",
  "preserve_comments": true,
  "optimize": false
}
```

**Supported Languages:** Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Solidity

### Light Encoding

**POST** `/api/v1/encode`

```json
{
  "text": "LUXBIN",
  "include_timing": true
}
```

**Response:**
```json
{
  "light_sequence": [
    {"character": "L", "wavelength_nm": 442.9, "color": "blue", "hex": "#0000ff"},
    {"character": "U", "wavelength_nm": 477.9, "color": "cyan", "hex": "#00ffff"}
  ],
  "total_duration_ms": 600
}
```

### Quantum Random

**POST** `/api/v1/quantum/random`

```json
{
  "count": 10,
  "min_value": 0,
  "max_value": 255,
  "format": "integers"
}
```

**Response:**
```json
{
  "values": [42, 187, 93, 12, 255, 0, 128, 64, 200, 31],
  "source": "ibm_quantum",
  "backend": "ibm_fez",
  "job_id": "abc123..."
}
```

---

## Pricing

| Tier | Price | Requests/Day | Quantum RNG |
|:-----|:-----:|:------------:|:-----------:|
| Free | $0 | 100 | Simulated |
| Pro | $29/mo | 10,000 | **Real IBM Quantum** |
| Enterprise | $299/mo | Unlimited | Dedicated Backend |

---

## Deployment

### Vercel

```bash
vercel deploy
```

### Docker

```bash
docker build -t luxbin-api .
docker run -p 8000:8000 --env-file .env luxbin-api
```

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

---

## LUXBIN Ecosystem

| Repository | Description |
|:-----------|:------------|
| [luxbin-quantum-grid](https://github.com/nichechristie/luxbin-quantum-grid) | Starlink smart grid integration |
| [Luxbin-Quantum-internet](https://github.com/nichechristie/Luxbin-Quantum-internet) | Full quantum internet stack |
| [LUXBIN_Light_Language-](https://github.com/nichechristie/LUXBIN_Light_Language-) | Photonic encoding protocol |

---

## License

MIT

## Author

**Nichole Christie** · [@nichechristie](https://github.com/nichechristie) · `luxbin.base.eth`
