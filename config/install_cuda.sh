#!/bin/bash

# CUDA Toolkit Installation Script
# Supports Ubuntu/Debian, CentOS/RHEL, and Docker setup
# Usage: ./install_cuda.sh [docker|ubuntu|centos|windows]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. Some operations may require sudo."
    fi
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="ubuntu"
        elif [ -f /etc/redhat-release ]; then
            OS="centos"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    print_status "Detected OS: $OS"
}

# Check NVIDIA driver
check_nvidia_driver() {
    print_status "Checking NVIDIA driver..."
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA driver found:"
        nvidia-smi
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        print_status "Driver Version: $DRIVER_VERSION"
        return 0
    else
        print_error "NVIDIA driver not found. Please install NVIDIA drivers first."
        print_status "Visit: https://www.nvidia.com/Download/index.aspx"
        return 1
    fi
}

# Install CUDA on Ubuntu/Debian
install_cuda_ubuntu() {
    print_header "Installing CUDA Toolkit on Ubuntu/Debian"
    
    # Check if CUDA already installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "CUDA already installed: $CUDA_VERSION"
        return 0
    fi
    
    # Get Ubuntu version
    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Ubuntu Version: $UBUNTU_VERSION"
    
    # Add NVIDIA repository
    print_status "Adding NVIDIA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update
    
    # Install CUDA Toolkit
    print_status "Installing CUDA Toolkit 12.3..."
    sudo apt-get -y install cuda-toolkit-12-3
    
    # Set environment variables
    print_status "Setting up environment variables..."
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Reload environment
    export PATH=/usr/local/cuda-12.3/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
    
    print_status "CUDA installation completed!"
    print_status "Please reboot your system and run: source ~/.bashrc"
}

# Install CUDA on CentOS/RHEL
install_cuda_centos() {
    print_header "Installing CUDA Toolkit on CentOS/RHEL"
    
    # Check if CUDA already installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "CUDA already installed: $CUDA_VERSION"
        return 0
    fi
    
    # Add NVIDIA repository
    print_status "Adding NVIDIA repository..."
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    
    # Install CUDA Toolkit
    print_status "Installing CUDA Toolkit 12.3..."
    sudo yum clean all
    sudo yum -y install cuda-toolkit-12-3
    
    # Set environment variables
    print_status "Setting up environment variables..."
    echo 'export PATH=/usr/local/cuda-12.3/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    print_status "CUDA installation completed!"
    print_status "Please reboot your system and run: source ~/.bashrc"
}

# Install NVIDIA Container Toolkit for Docker
install_docker_cuda() {
    print_header "Installing NVIDIA Container Toolkit for Docker"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    # Detect distribution
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    print_status "Distribution: $distribution"
    
    # Add repository
    print_status "Adding NVIDIA Docker repository..."
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Install nvidia-docker2
    print_status "Installing nvidia-docker2..."
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    
    # Restart Docker service
    print_status "Restarting Docker service..."
    sudo systemctl restart docker
    
    # Test GPU access in Docker
    print_status "Testing GPU access in Docker..."
    docker run --rm --gpus all nvidia/cuda:12.3-base-ubuntu22.04 nvidia-smi
    
    print_status "NVIDIA Container Toolkit installed successfully!"
}

# Verify CUDA installation
verify_cuda() {
    print_header "Verifying CUDA Installation"
    
    # Check nvcc
    if command -v nvcc &> /dev/null; then
        print_status "NVCC version:"
        nvcc --version
    else
        print_warning "NVCC not found in PATH"
    fi
    
    # Check nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA-SMI:"
        nvidia-smi
    else
        print_warning "nvidia-smi not found"
    fi
    
    # Test CUDA compilation (if nvcc available)
    if command -v nvcc &> /dev/null; then
        print_status "Testing CUDA compilation..."
        cat > /tmp/test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("Found %d CUDA devices\n", count);
    return 0;
}
EOF
        
        nvcc /tmp/test_cuda.cu -o /tmp/test_cuda
        if [ -f /tmp/test_cuda ]; then
            /tmp/test_cuda
            rm /tmp/test_cuda /tmp/test_cuda.cu
            print_status "CUDA compilation test passed!"
        else
            print_error "CUDA compilation test failed!"
        fi
    fi
}

# Test Python CUDA support
test_python_cuda() {
    print_header "Testing Python CUDA Support"
    
    # Test PyTorch
    if python3 -c "import torch" 2>/dev/null; then
        print_status "Testing PyTorch CUDA support:"
        python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
    else
        print_warning "PyTorch not installed"
    fi
    
    # Test TensorFlow
    if python3 -c "import tensorflow" 2>/dev/null; then
        print_status "Testing TensorFlow GPU support:"
        python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpu_devices = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpu_devices)}')
for gpu in gpu_devices:
    print(f'GPU: {gpu}')
"
    else
        print_warning "TensorFlow not installed"
    fi
}

# Windows installation instructions
show_windows_instructions() {
    print_header "Windows CUDA Installation"
    print_status "Please follow these steps manually:"
    echo ""
    echo "1. Download CUDA Toolkit from:"
    echo "   https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "2. Select: Windows → x86_64 → Your Windows Version → exe (local)"
    echo ""
    echo "3. Run installer as Administrator"
    echo "   - Accept EULA"
    echo "   - Select CUDA Toolkit"
    echo "   - Select Visual Studio Integration (if using VS)"
    echo "   - CUDA Samples (optional)"
    echo ""
    echo "4. Verify installation:"
    echo "   - Open Command Prompt as Administrator"
    echo "   - Run: nvcc --version"
    echo "   - Run: nvidia-smi"
    echo ""
    echo "5. For Docker on Windows:"
    echo "   - Install Docker Desktop with WSL2 backend"
    echo "   - Enable GPU support in Docker Desktop settings"
}

# Main installation function
main() {
    print_header "CUDA Toolkit Installation Script"
    
    # Parse command line arguments
    INSTALL_TYPE=${1:-auto}
    
    case $INSTALL_TYPE in
        "docker")
            check_root
            check_nvidia_driver
            install_docker_cuda
            ;;
        "ubuntu")
            check_root
            check_nvidia_driver
            install_cuda_ubuntu
            verify_cuda
            ;;
        "centos")
            check_root
            check_nvidia_driver
            install_cuda_centos
            verify_cuda
            ;;
        "windows")
            show_windows_instructions
            ;;
        "auto")
            detect_os
            case $OS in
                "ubuntu")
                    check_root
                    check_nvidia_driver
                    install_cuda_ubuntu
                    verify_cuda
                    ;;
                "centos")
                    check_root
                    check_nvidia_driver
                    install_cuda_centos
                    verify_cuda
                    ;;
                "windows")
                    show_windows_instructions
                    ;;
                *)
                    print_error "Unsupported OS: $OS"
                    print_status "Available options: docker, ubuntu, centos, windows"
                    exit 1
                    ;;
            esac
            ;;
        "verify")
            verify_cuda
            test_python_cuda
            ;;
        *)
            echo "Usage: $0 [docker|ubuntu|centos|windows|verify|auto]"
            echo ""
            echo "Options:"
            echo "  docker   - Install NVIDIA Container Toolkit for Docker"
            echo "  ubuntu   - Install CUDA Toolkit on Ubuntu/Debian"
            echo "  centos   - Install CUDA Toolkit on CentOS/RHEL"
            echo "  windows  - Show Windows installation instructions"
            echo "  verify   - Verify existing CUDA installation"
            echo "  auto     - Auto-detect OS and install (default)"
            exit 1
            ;;
    esac
    
    # Test Python CUDA support if CUDA was installed
    if [[ "$INSTALL_TYPE" == "ubuntu" || "$INSTALL_TYPE" == "centos" || "$INSTALL_TYPE" == "auto" ]]; then
        test_python_cuda
    fi
    
    print_header "Installation Complete!"
    print_status "For Ray cluster setup, ensure all nodes have CUDA installed."
    print_status "For Docker-based setup, use: $0 docker"
}

# Run main function with all arguments
main "$@"