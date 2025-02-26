import React, { useState } from "react";
import ImageDisplay from "./components/ImageDisplay";
import CameraFeed from "./components/CameraFeed";


const App = () => {
  const [currentWord, setCurrentWord] = useState("");
  const [recognizedSigns, setRecognizedSigns] = useState([]);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Tamil Sign Language Trainer</h1>
      <div style={{ display: "flex", justifyContent: "center", gap: "20px" }}>
        <ImageDisplay 
          setCurrentWord={setCurrentWord} 
          recognizedSigns={recognizedSigns} 
          setRecognizedSigns={setRecognizedSigns} 
        />
        <CameraFeed onCapture={(sign) => setRecognizedSigns([...recognizedSigns, sign])} />
      </div>
    </div>
  );
};

export default App;
