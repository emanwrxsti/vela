# Docker deployment

`ocminersupr/supr-meow-vela:0.1.0` (also `:latest`). Install the NVIDIA Container
Toolkit on the host. Keep a persistent model cache; the ~1.2 GB sd-turbo-int
model downloads automatically on first start (default pinned, checksum-verified).

```sh
docker run -d --name vela --gpus '"device=0"' --restart unless-stopped \
  -v vela-models:/models \
  -e WALLET='vela1…YOUR_ADDRESS' -e WORKER=rig1 \
  ocminersupr/supr-meow-vela:0.1.0
```

`POOL_URL` defaults to `vela.suprnova.cc:7601`. `PASSWORD` defaults to `x`.
`MODEL_DIR` defaults to `/models`; `MODEL_PATH` selects an existing copy.
Larger GPUs can set `BATCH=8 LANES=2`; multi-GPU uses `--gpus '"device=0,1"'`
and `DEVICES=0,1`. TLS is not built into the image — use the plain-TCP port.

On rented/headless rigs, set `SSH_PUBKEY` (preferred) or `SSH_PASSWORD` to get
a shell into the container (optionally `SSH_PORT`).
