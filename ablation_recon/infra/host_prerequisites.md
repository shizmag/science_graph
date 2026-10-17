# Host prerequisites

Linux: RTX-class GPU, current NVIDIA driver, Docker/Compose and NVIDIA Container Toolkit. PyTorch CUDA13 metadata lists Linux driver minimum 580.65.06 for inspected release; recheck. GPU must be visible inside container. Admin/root setup/reboot may be needed; no automatic driver installation. Toolkit guide: [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Windows 11 initial target: current NVIDIA WSL driver (PyTorch 2.12 CUDA13 release metadata lists Windows minimum 580.88), WSL2/current kernel, Docker Desktop Linux containers/GPU support, sufficient disk and WSL ext4/named volume. Enablement/admin/reboot may be required. PowerShell can invoke WSL wrapper but cannot install host dependencies automatically. Windows 10 21H2+ is documented conditionally, not tested.

Future preflight reports OS/architecture, Docker, driver/GPU, container GPU visibility, free disk, WSL/kernel and writable paths. Never print credentials.
