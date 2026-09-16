"""Team GPU check: Python 3.14.4, CUDA >= 12.8, RTX 5080 sm_120."""
import sys


def main():
    if sys.version_info[:3] != (3, 14, 4):
        raise RuntimeError(f"팀 Python 버전은 3.14.4입니다. 현재: {sys.version.split()[0]}")
    import torch
    print("torch", torch.__version__, "built CUDA", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가: Windows 드라이버와 cu128 PyTorch 설치를 확인하세요.")
    cuda = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
    if cuda < (12, 8):
        raise RuntimeError("팀 기준은 CUDA 12.8 이상 빌드입니다.")
    print("gpu", torch.cuda.get_device_name(0))
    print("arch", torch.cuda.get_arch_list())
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("팀 RTX 5080용 sm_120 지원 빌드를 확인하세요.")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("ok", y.shape)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        sys.exit(1)
