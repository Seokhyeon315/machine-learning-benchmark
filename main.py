import torch

def main():
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print ("MPS is available")
    else:
        device = torch.device("cpu")
        print ("MPS is not available, using CPU instead")

if __name__ == "__main__":
    main()
