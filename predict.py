from inference import DeepfakeDetector
import os


def main():
    # Initialize detector
    checkpoint_path = os.path.join("checkpoints", "best_model.pth")
    print(f"Initializing detector with checkpoint: {checkpoint_path}")
    detector = DeepfakeDetector(checkpoint_path)

    # Path to video
    video_path = os.path.join("data", "videos", "test", "id17_2323.mp4")
    print(f"Processing video: {video_path}")

    try:
        score, confidence = detector.predict_video(video_path)

        print("\nResults:")
        print(f"Video: {video_path}")
        print(f"Manipulation Score: {score:.4f}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Prediction: {'FAKE' if score > 0.5 else 'REAL'}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()