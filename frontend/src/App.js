import React, { useState } from 'react';
import './App.css';
import VideoGenerator from './components/VideoGenerator';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Video Generator</h1>
        <p>Transform photos, text, or videos into stunning AI-generated videos</p>
      </header>
      <main>
        <VideoGenerator />
      </main>
    </div>
  );
}

export default App;

