from PIL import Image
import argparse
import os
from decoder import Decoder
from encoder import Encoder

def main():
    parser = argparse.ArgumentParser(description="Encode or decode an image.")
    parser.add_argument("--encode", type=str, help="Run encoder for this image")
    parser.add_argument("--decode", type=str, help="Run decoder for the transformations")
    parser.add_argument("--output", type=str, help="Output path for encoded transformations or decoded image")
    parser.add_argument("--iterations", type=int, default=6, help="Number of iterations for decoding")
    parser.add_argument("--error_threshold", type=float, default=0.01, help="Error threshold for encoding")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.encode:
        if not args.output:
            raise ValueError("Output path is required for encoding.")
        # if output already exists, raise error
        if os.path.exists(args.output):
            raise ValueError("Output path already exists.")
        
        print("Encoding...")
        encoder = Encoder(args.encode, verbose=args.verbose, error_threshold=args.error_threshold)
        transforms = encoder.encode()
        encoder.save_transforms_to_csv(transforms, args.output)
    
    elif args.decode:
        if not args.output:
            raise ValueError("Output path is required for decoding.")
        # if output already exists, raise error
        if os.path.exists(args.output):
            raise ValueError("Output path already exists.")
        
        print("Decoding...")
        decoder = Decoder((256, 256), iterations=args.iterations)
        decoder.load_from_csv(args.decode)
        decoded_image = decoder.decode()
        Image.fromarray((decoded_image * 255).astype('uint8')).save(args.output)
        
if __name__ == "__main__":
    main()
