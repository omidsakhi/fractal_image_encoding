from PIL import Image
import argparse
import os
from decoder import Decoder
from encoder import Encoder

def main():
    parser = argparse.ArgumentParser(description="Encode or decode an image.")
    parser.add_argument("--encode", type=str, help="Run encoder for this image")
    parser.add_argument("--decode", type=str, help="Path to transforms JSON (from encoding)")
    parser.add_argument("--output", type=str, help="Output path for encoded JSON or decoded image")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for decoding")
    parser.add_argument("--error_threshold", type=float, default=0.001, help="Error threshold for encoding")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for the encoder (1 = serial)")
    parser.add_argument("--image_size", type=str, default=None, help="Decoder canvas as HxW (overrides image_height/width in JSON)")
    parser.add_argument("--save_iterations", action="store_true", help="Save an image for every decoder iteration (iteration 0 = seed)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.encode:
        if not args.output:
            raise ValueError("Output path is required for encoding.")
        # if output already exists, raise error
        if os.path.exists(args.output):
            raise ValueError("Output path already exists.")
        
        print("Encoding...")
        encoder = Encoder(args.encode, verbose=args.verbose, error_threshold=args.error_threshold, num_workers=args.workers)
        transforms = encoder.encode()
        encoder.save_transforms(transforms, args.output)
    
    elif args.decode:
        if not args.output:
            raise ValueError("Output path is required for decoding.")
        # if output already exists, raise error
        if os.path.exists(args.output):
            raise ValueError("Output path already exists.")
        
        print("Decoding...")
        image_size = None
        if args.image_size:
            h, w = args.image_size.lower().split('x')
            image_size = (int(h), int(w))
        decoder = Decoder(args.decode, image_size=image_size, iterations=args.iterations)

        on_iteration = None
        if args.save_iterations:
            stem, ext = os.path.splitext(args.output)
            def on_iteration(iteration, image):
                path = f"{stem}_iter{iteration:03d}{ext}"
                Image.fromarray((image * 255).astype('uint8')).save(path)

        decoded_image = decoder.decode(on_iteration=on_iteration)
        Image.fromarray((decoded_image * 255).astype('uint8')).save(args.output)
        
if __name__ == "__main__":
    main()
