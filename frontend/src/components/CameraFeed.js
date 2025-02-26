import React, { useEffect, useRef, useState } from 'react';

const CameraFeed = ({ onCapture }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const processingRef = useRef(false);
  const hasDetectedSignRef = useRef(false);  // Track if we've already captured a sign

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: 640,
            height: 480,
            facingMode: 'user'
          } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    let frameId;

    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current || processingRef.current) {
        frameId = requestAnimationFrame(processFrame);
        return;
      }

      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      
      context.clearRect(0, 0, canvas.width, canvas.height);
      
      context.save();
      context.scale(-1, 1);
      context.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
      context.restore();

      // Only process frames if we're capturing and haven't detected a sign yet
      if (isCapturing && !hasDetectedSignRef.current) {
        try {
          processingRef.current = true;
          
          const boxSize = 300;
          const x = (canvas.width - boxSize) / 2;
          const y = (canvas.height - boxSize) / 2;
          const frameData = context.getImageData(x, y, boxSize, boxSize);
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = boxSize;
          tempCanvas.height = boxSize;
          const tempContext = tempCanvas.getContext('2d');
          tempContext.putImageData(frameData, 0, 0);

          const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg'));
          const formData = new FormData();
          formData.append('frame', blob);

          const response = await fetch('http://localhost:5000/process-frame', {
            method: 'POST',
            body: formData
          });

          const result = await response.json();
          
          if (result.letter && result.confidence > 0.7) {
            hasDetectedSignRef.current = true;  // Mark that we've detected a sign
            onCapture(result.letter);
            setIsCapturing(false);  // Automatically stop capturing after detection
          }
        } catch (error) {
          console.error('Error processing frame:', error);
        } finally {
          processingRef.current = false;
        }
      }

      const boxSize = 300;
      const x = (canvas.width - boxSize) / 2;
      const y = (canvas.height - boxSize) / 2;
      
      context.strokeStyle = isCapturing ? '#00ff00' : '#ff0000';
      context.lineWidth = 3;
      context.strokeRect(x, y, boxSize, boxSize);

      context.fillStyle = 'white';
      context.font = '20px Arial';
      context.textAlign = 'center';
      const text = isCapturing ? 'Show sign in the box' : 'Click Start to begin';
      context.fillText(text, canvas.width / 2, y - 10);

      frameId = requestAnimationFrame(processFrame);
    };

    frameId = requestAnimationFrame(processFrame);

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
  }, [isCapturing, onCapture]);

  const toggleCapture = () => {
    if (!isCapturing) {
      // Starting new capture
      hasDetectedSignRef.current = false;  // Reset the detection flag
      setIsCapturing(true);
    } else {
      // Stopping capture
      setIsCapturing(false);
    }
  };

  return (
    <div style={{ 
      position: 'relative',
      width: '640px',
      height: '480px',
      border: '2px solid #ccc',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{
          width: '100%',
          height: '100%',
          transform: 'scaleX(-1)',
          display: 'none'
        }}
      />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        style={{
          width: '100%',
          height: '100%'
        }}
      />
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 2
      }}>
        <button
          onClick={toggleCapture}
          style={{
            padding: '10px 20px',
            fontSize: '1.1rem',
            backgroundColor: isCapturing ? '#ff4444' : '#44aa44',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          {isCapturing ? 'Stop Capture' : 'Start Capture'}
        </button>
      </div>
    </div>
  );
};
export default CameraFeed;
