import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setResponse("");
    try {
      const res = await axios.post("http://localhost:8000/generate-recipe", {
        query,
      });
      setResponse(res.data.response);
    } catch (err) {
      setResponse("An error occurred: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>🍳 AI Recipe Assistant</h1>
      <input
        type="text"
        placeholder="What can I cook with tomatoes and eggs?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSearch} disabled={loading}>
        {loading ? "Generating..." : "Find Recipes"}
      </button>
      <div className="response">
        {response && <div className="response">{response}</div>}
      </div>
    </div>
  );
}

export default App;
