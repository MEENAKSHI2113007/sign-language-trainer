const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();

// Enable CORS
app.use(cors());

// Define images directory path
const imagesDir = path.join(__dirname, 'images');

// Serve images directly from the images directory
app.use('/images', express.static(imagesDir));

// Load and validate words data
let words;
try {
  words = require('./words.json');
  
  // Validate words.json structure
  if (!Array.isArray(words) || words.length === 0) {
    console.error("❌ words.json must contain a non-empty array!");
    process.exit(1);
  }

  // Log available words
  console.log(`✅ Loaded ${words.length} words from words.json`);
} catch (error) {
  console.error("❌ Error loading words.json:", error);
  process.exit(1);
}

app.get('/word', (req, res) => {
  try {
    const randomWord = words[Math.floor(Math.random() * words.length)];
    
    // Construct the full image URL
    const baseUrl = `${req.protocol}://${req.get('host')}`;
    const wordData = {
      ...randomWord,
      image: `${baseUrl}${randomWord.image}`
    };
    
    // Log successful request
    console.log(`✅ Serving word: ${wordData.word}, image: ${wordData.image}`);
    
    res.json(wordData);
  } catch (error) {
    console.error('❌ Error:', error);
    res.status(500).json({ error: 'Failed to fetch word' });
  }
});

// Add a health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 5001;

app.listen(PORT, () => {
  console.log(`✅ Express server running on port ${PORT}`);
  console.log(`✅ Serving images from: ${imagesDir}`);
  
  // List all files in images directory
  try {
    const files = fs.readdirSync(imagesDir);
    console.log("\n✅ Available images:");
    files.forEach(file => console.log(`   - ${file}`));
  } catch (error) {
    console.error("❌ Error reading images directory:", error);
  }
});

// Handle errors globally
app.use((err, req, res, next) => {
  console.error('❌ Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});
