import React, { useEffect, useState, useRef } from "react";
import Confetti from "./Confetti";

const ImageDisplay = ({ setCurrentWord, recognizedSigns, setRecognizedSigns }) => {
  const [imageUrl, setImageUrl] = useState("");
  const [word, setWord] = useState("");
  const [characters, setCharacters] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [imageLoading, setImageLoading] = useState(true);
  const [timer, setTimer] = useState(30);
  const [feedback, setFeedback] = useState("");
  
  // Animation state
  const [animation, setAnimation] = useState({
    active: false,
    type: 'success',
    position: { x: 0, y: 0 }
  });
  
  // Animation counters
  const animationCountRef = useRef({
    success: 0,
    failure: 0
  });
  
  // Reference to current word
  const currentWordRef = useRef("");

  const fetchWord = async () => {
    // Immediately stop any ongoing animation when fetching a new word
    setAnimation({
      active: false,
      type: 'success',
      position: { x: 0, y: 0 }
    });
    
    setLoading(true);
    setImageLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:5001/word");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      if (!data.image || !data.word || !data.characters) {
        throw new Error("Invalid data received from server");
      }

      setImageUrl(data.image);
      setWord(data.word);
      setCharacters(data.characters);
      setCurrentWord(data.word);
      setRecognizedSigns([]);
      setTimer(30);
      setFeedback("");
      
      // Reset animation counters when word changes
      if (currentWordRef.current !== data.word) {
        currentWordRef.current = data.word;
        animationCountRef.current = {
          success: 0,
          failure: 0
        };
      }
    } catch (error) {
      setError(`Failed to fetch word: ${error.message}`);
      console.error("Error fetching word:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWord();

    const wordInterval = setInterval(() => {
      fetchWord();
    }, 30000);

    const countdown = setInterval(() => {
      setTimer((prev) => {
        if (prev <= 1) {
          // Reset animation when timer runs out
          setAnimation({
            active: false,
            type: 'success',
            position: { x: 0, y: 0 }
          });
          return 30;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      clearInterval(wordInterval);
      clearInterval(countdown);
    };
  }, [setCurrentWord, setRecognizedSigns]);

  const handleImageLoad = () => {
    setImageLoading(false);
  };

  const handleImageError = () => {
    setError("Failed to load image");
    setImageLoading(false);
  };

  const handleClear = () => {
    setRecognizedSigns([]);
    setFeedback("");
  };

  const handleBackspace = () => {
    if (recognizedSigns.length > 0) {
      setRecognizedSigns(recognizedSigns.slice(0, -1));
    }
  };

  const handleSubmit = () => {
    const recognizedText = recognizedSigns.join("");
    const isCorrect = recognizedText === word;
    
    setFeedback(
      isCorrect ? "✅ Correct!" : `❌ Incorrect! Correct word: ${word}`
    );
    
    // Get the center position of the image container
    const imageContainer = document.querySelector('.image-container');
    const position = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    
    if (imageContainer) {
      const rect = imageContainer.getBoundingClientRect();
      position.x = rect.left + rect.width / 2;
      position.y = rect.top + rect.height / 2;
    }
    
    // Determine animation type
    const animationType = isCorrect ? 'success' : 'failure';
    
    // Check if we've shown this animation type fewer than 3 times for this word
    if (animationCountRef.current[animationType] < 3) {
      // Increment the counter
      animationCountRef.current[animationType]++;
      
      // Show the animation
      setAnimation({
        active: true,
        type: animationType,
        position
      });

      // Play appropriate sound
      if (isCorrect) {
        const audio = new Audio('/success-sound.mp3');
        audio.volume = 0.5;
        audio.play().catch(e => console.log('Audio play failed:', e));
      }
    }
  };

  const handleSkip = () => {
    // Explicitly disable animation before fetching a new word
    setAnimation({
      active: false,
      type: 'success',
      position: { x: 0, y: 0 }
    });
    fetchWord();
  };
  
  const handleAnimationComplete = () => {
    setAnimation({ ...animation, active: false });
  };

  if (loading) return <div>Loading word data...</div>;
  if (error) return <div style={{ color: "red" }}>{error}</div>;

  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          fontSize: "1.5rem",
          fontWeight: "bold",
          color: timer <= 5 ? "red" : "black",
        }}
      >
        Time left: {timer}s
      </div>
      <div
        className="image-container"
        style={{
          width: "300px",
          height: "300px",
          margin: "20px auto",
          overflow: "hidden",
          position: "relative",
          backgroundColor: "#f0f0f0",
          border: "1px solid #ddd",
          borderRadius: "8px",
        }}
      >
        {imageLoading && <div>Loading image...</div>}
        <img
          src={imageUrl}
          alt="Sign Prompt"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: imageLoading ? "none" : "block",
          }}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      </div>
      <p style={{ fontSize: "1.5rem" }}>{characters.map(() => "_ ").join("")}</p>
      <p style={{ fontSize: "1.2rem", color: "blue" }}>
        Your signs: {recognizedSigns.join("")}
      </p>

      <div style={{ marginTop: "10px" }}>
        <button
          onClick={handleClear}
          style={{ margin: "5px", padding: "10px", fontSize: "1rem" }}
        >
          Clear
        </button>
        <button
          onClick={handleBackspace}
          style={{ margin: "5px", padding: "10px", fontSize: "1rem" }}
        >
          Backspace
        </button>
        <button
          onClick={handleSubmit}
          style={{ margin: "5px", padding: "10px", fontSize: "1rem" }}
        >
          Submit
        </button>
        <button
          onClick={handleSkip}
          style={{ margin: "5px", padding: "10px", fontSize: "1rem" }}
        >
          Skip
        </button>
      </div>

      {feedback && (
        <p
          style={{
            fontSize: "1.2rem",
            color: feedback.includes("❌") ? "red" : "green",
          }}
        >
          {feedback}
        </p>
      )}
      
      <Confetti 
        active={animation.active} 
        x={animation.position.x}
        y={animation.position.y}
        type={animation.type}
        onComplete={handleAnimationComplete}
      />
    </div>
  );
};

export default ImageDisplay;
