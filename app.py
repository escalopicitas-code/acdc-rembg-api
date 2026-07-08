# ACDC CarveKit API

API de remoção de fundo para o sistema ACDC Casa.
Usa CarveKit com modelo tracer_b7, otimizado para produtos e móveis.

## Deploy automático
Push na branch `main` → GitHub Actions builda no Azure Container Registry → deploy no Azure Container Apps.

## Endpoint

`POST /remove-background`

```json
{ "image": "data:image/jpeg;base64,..." }
```

Retorna PNG 1024x1024 com fundo transparente e produto centralizado.
