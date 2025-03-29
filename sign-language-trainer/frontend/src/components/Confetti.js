import React, { useEffect, useRef } from 'react';

const Confetti = ({ active, x, y, onComplete, type = 'success' }) => {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationRef = useRef(null);
  
  useEffect(() => {
    if (!active) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    if (type === 'success') {
      // SUCCESS ANIMATION: CONFETTI
      const particles = [];
      const colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#6B88FE', '#FF85A1'];
      
      // Create particles
      for (let i = 0; i < 100; i++) {
        particles.push({
          x: x || canvas.width / 2,
          y: y || canvas.height / 2,
          size: Math.random() * 5 + 2,
          color: colors[Math.floor(Math.random() * colors.length)],
          vx: (Math.random() - 0.5) * 12,
          vy: (Math.random() - 0.5) * 12 - 3,
          gravity: 0.1 + Math.random() * 0.1,
          opacity: 1,
          rotation: Math.random() * 360,
          rotationSpeed: (Math.random() - 0.5) * 10
        });
      }
      particlesRef.current = particles;
      
      // Update and draw particles
      const animate = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        let allDone = true;
        
        particlesRef.current.forEach((p, index) => {
          p.x += p.vx;
          p.y += p.vy;
          p.vy += p.gravity;
          p.opacity -= 0.01;
          p.rotation += p.rotationSpeed;
          
          if (p.opacity > 0) {
            allDone = false;
            
            ctx.save();
            ctx.globalAlpha = p.opacity;
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rotation * Math.PI / 180);
            
            // Draw a simple shape (square, circle, or star)
            const shapeType = index % 3;
            ctx.fillStyle = p.color;
            
            if (shapeType === 0) {
              // Square
              ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size);
            } else if (shapeType === 1) {
              // Circle
              ctx.beginPath();
              ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
              ctx.fill();
            } else {
              // Star
              const spikes = 5;
              const outerRadius = p.size;
              const innerRadius = p.size / 2;
              
              ctx.beginPath();
              for (let i = 0; i < spikes * 2; i++) {
                const radius = i % 2 === 0 ? outerRadius : innerRadius;
                const angle = Math.PI * i / spikes;
                ctx.lineTo(Math.cos(angle) * radius, Math.sin(angle) * radius);
              }
              ctx.closePath();
              ctx.fill();
            }
            
            ctx.restore();
          }
        });
        
        if (allDone) {
          cancelAnimationFrame(animationRef.current);
          if (onComplete) onComplete();
        } else {
          animationRef.current = requestAnimationFrame(animate);
        }
      };
      
      animate();
    } else if (type === 'failure') {
      // FAILURE ANIMATION: CROSS MARK (X) instead of thumbs down
      let scale = 0;
      let opacity = 0;
      const maxScale = 1;
      const centerX = x || canvas.width / 2;
      const centerY = y || canvas.height / 2;
      
      const animate = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Increase scale and opacity
        if (scale < maxScale) {
          scale += 0.05;
          opacity = Math.min(1, opacity + 0.1);
        } else {
          // Once fully scaled, start fading out
          opacity -= 0.02;
        }
        
        if (opacity <= 0) {
          cancelAnimationFrame(animationRef.current);
          if (onComplete) onComplete();
          return;
        }
        
        ctx.save();
        ctx.globalAlpha = opacity;
        ctx.translate(centerX, centerY);
        ctx.scale(scale, scale);
        
        // Draw X mark (cross)
        ctx.strokeStyle = '#FF4136';
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        
        // Draw X
        ctx.beginPath();
        ctx.moveTo(-25, -25);
        ctx.lineTo(25, 25);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(25, -25);
        ctx.lineTo(-25, 25);
        ctx.stroke();
        
        ctx.restore();
        
        animationRef.current = requestAnimationFrame(animate);
      };
      
      animate();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [active, x, y, onComplete, type]);
  
  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 9999
      }}
    />
  );
};

export default Confetti;
