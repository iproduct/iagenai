Open Docker Desktop.Go to Settings (Gear icon) -> Docker Engine.Replace your JSON configuration with this exact structure. 
This forces containerd to behave like a standard sequential HTTP downloader, preventing it from overwhelming either your host OS or the standard NAT engine:
```json
{
  "features": {
    "containerd-snapshotter": true
  },
  "max-concurrent-downloads": 1,
  "max-concurrent-uploads": 1,
  "env": [
    "GODEBUG=http2client=0"
  ]
}
```