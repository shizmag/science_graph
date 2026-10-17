# Windows and Linux support

| Host | Shared execution | Prerequisites | Status |
|---|---|---|---|
| Ubuntu/Linux x86_64 | pinned Linux CUDA OCI | NVIDIA driver, Docker/Compose, NVIDIA Container Toolkit, GPU visibility | primary, metadata only |
| Windows 11 WSL2 | same Linux OCI via Docker Desktop | current NVIDIA WSL driver, WSL2/kernel, Docker Desktop GPU, ext4/named volume | documented, untested |
| Windows 10 21H2+ WSL2 | same image if versions support forwarding | compatible driver/kernel/Docker Desktop | conditional, untested |
| Native Windows | none established | separate work | unsupported initially |

Docker GPU docs: [Docker Desktop GPU documentation](https://docs.docker.com/desktop/features/gpu/). Microsoft WSL CUDA: [Microsoft CUDA in WSL documentation](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl). Docker WSL guidance: [Docker WSL documentation](https://docs.docker.com/desktop/features/wsl/) and [Docker WSL documentation](https://docs.docker.com/desktop/features/wsl/)best-practices/. Vendor support does not test this project.

One command is feasible after host prerequisites: Linux shell or PowerShell wrapper invoking WSL and same image. Bootstrap cannot install drivers, enable WSL/virtualization, administer Docker or reboot host. Keep large files in WSL ext4 rather than /mnt/c. No cross-host test occurred.
