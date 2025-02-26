import React, { useEffect, useState } from "react";

const ImageDisplay = ({ setCurrentWord, recognizedSigns, setRecognizedSigns }) => {
  const [imageUrl, setImageUrl] = useState("");
  const [word, setWord] = useState("");
  const [characters, setCharacters] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [imageLoading, setImageLoading] = useState(true);
  const [timer, setTimer] = useState(30);
  const [feedback, setFeedback] = useState("");

  const fetchWord = async () => {
    setLoading(true);
    setImageLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:5001/word");
      if (!response.ok) {
        throw new Error(HTTP error! status: ${response.status});
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
    } catch (error) {
      setError(Failed to fetch word: ${error.message});
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
      setTimer((prev) => (prev > 0 ? prev - 1 : 30));
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
    setRecognizedSigns([]); // Clear the recognized signs array
    setFeedback(""); // Clear any feedback
  };

  const handleSubmit = () => {
    const recognizedText = recognizedSigns.join("");
    setFeedback(
      recognizedText === word ? "✅ Correct!" : ❌ Incorrect! Correct word: ${word}
    );
  };

  const handleSkip = () => {
    fetchWord();
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
    </div>
  );
};

export default ImageDisplay;
