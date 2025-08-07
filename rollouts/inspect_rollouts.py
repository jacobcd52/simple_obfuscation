#!/usr/bin/env python3
"""
Rollout Inspector - Web Interface

This script creates a web interface for inspecting rollout files.
Run this script to start a local web server for browsing rollouts.
"""

import json
import os
import glob
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

def get_rollout_files():
    """Get all rollout JSON files sorted by number."""
    rollout_dir = Path(__file__).parent
    json_files = glob.glob(str(rollout_dir / "rollout_*.json"))
    
    # Extract numbers and sort
    rollouts = []
    for file_path in json_files:
        filename = os.path.basename(file_path)
        if filename.startswith("rollout_") and filename.endswith(".json"):
            try:
                number = int(filename[8:-5])  # Extract number from "rollout_00000000.json"
                rollouts.append((number, file_path))
            except ValueError:
                continue
    
    return sorted(rollouts)

def load_rollout(file_path):
    """Load a rollout file and return its data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load rollout: {e}"}

def format_rewards(reward_breakdown, total_reward):
    """Format rewards for display."""
    if not reward_breakdown:
        return f"Total: {total_reward:.1f}"
    
    parts = []
    for key, value in reward_breakdown.items():
        parts.append(f"{key}: {value:.1f}")
    parts.append(f"Total: {total_reward:.1f}")
    return " | ".join(parts)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Rollout Inspector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .controls {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .input-group {
            display: inline-block;
            margin-right: 10px;
        }
        input[type="number"] {
            width: 100px;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            padding: 8px 16px;
            margin: 0 5px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .rollout-info {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .rollout-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .rollout-rewards {
            font-size: 14px;
            color: #666;
        }
        .content-section {
            margin-bottom: 20px;
        }
        .content-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .content-text {
            white-space: pre-wrap;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
        }
        .error {
            color: red;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        .stats {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rollout Inspector</h1>
        
        <div class="controls">
            <div class="input-group">
                <label for="rolloutNumber">Rollout Number:</label>
                <input type="number" id="rolloutNumber" value="{{ current_rollout }}" min="0" max="{{ max_rollout }}">
            </div>
            <button onclick="loadRollout()">Load</button>
            <button onclick="previousRollout()">Previous</button>
            <button onclick="nextRollout()">Next</button>
            <button onclick="skipRollout(10)">+10</button>
            <button onclick="skipRollout(100)">+100</button>
        </div>
        
        <div id="rolloutContent">
            <div class="loading">Loading rollout {{ current_rollout }}...</div>
        </div>
    </div>
    
    <script>
        const rollouts = {{ rollouts_json | safe }};
        let currentIndex = {{ current_index }};
        
        function findRolloutIndex(number) {
            return rollouts.indexOf(number);
        }
        
        function loadRollout() {
            const number = parseInt(document.getElementById('rolloutNumber').value);
            if (rollouts.includes(number)) {
                currentIndex = findRolloutIndex(number);
                fetchRollout(number);
            } else {
                document.getElementById('rolloutContent').innerHTML = 
                    '<div class="error">Rollout ' + number + ' not found.</div>';
            }
        }
        
        function previousRollout() {
            if (currentIndex > 0) {
                currentIndex--;
                const number = rollouts[currentIndex];
                document.getElementById('rolloutNumber').value = number;
                fetchRollout(number);
            }
        }
        
        function nextRollout() {
            if (currentIndex < rollouts.length - 1) {
                currentIndex++;
                const number = rollouts[currentIndex];
                document.getElementById('rolloutNumber').value = number;
                fetchRollout(number);
            }
        }
        
        function skipRollout(increment) {
            const newIndex = currentIndex + increment;
            if (newIndex >= 0 && newIndex < rollouts.length) {
                currentIndex = newIndex;
                const number = rollouts[currentIndex];
                document.getElementById('rolloutNumber').value = number;
                fetchRollout(number);
            }
        }
        
        function fetchRollout(number) {
            document.getElementById('rolloutContent').innerHTML = 
                '<div class="loading">Loading rollout ' + number + '...</div>';
            
            fetch('/api/rollout/' + number)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('rolloutContent').innerHTML = 
                            '<div class="error">' + data.error + '</div>';
                    } else {
                        document.getElementById('rolloutContent').innerHTML = data.html;
                    }
                })
                .catch(error => {
                    document.getElementById('rolloutContent').innerHTML = 
                        '<div class="error">Failed to load rollout: ' + error + '</div>';
                });
        }
        
        // Handle Enter key in input
        document.getElementById('rolloutNumber').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                loadRollout();
            }
        });
        
        // Load initial rollout
        window.onload = function() {
            fetchRollout({{ current_rollout }});
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page."""
    rollouts = get_rollout_files()
    
    if not rollouts:
        return "No rollout files found."
    
    # Get current rollout from URL parameter or default to first
    current_rollout = request.args.get('rollout', type=int)
    if current_rollout is None or not any(num == current_rollout for num, _ in rollouts):
        current_rollout = rollouts[0][0]
    
    # Find current index
    current_index = next(i for i, (num, _) in enumerate(rollouts) if num == current_rollout)
    
    return render_template_string(
        HTML_TEMPLATE,
        rollouts_json=json.dumps([num for num, _ in rollouts]),
        current_rollout=current_rollout,
        current_index=current_index,
        max_rollout=rollouts[-1][0] if rollouts else 0
    )

@app.route('/api/rollout/<int:rollout_number>')
def get_rollout(rollout_number):
    """API endpoint to get rollout data."""
    rollouts = get_rollout_files()
    
    # Find the rollout file
    rollout_file = None
    for num, file_path in rollouts:
        if num == rollout_number:
            rollout_file = file_path
            break
    
    if not rollout_file:
        return jsonify({"error": f"Rollout {rollout_number} not found"})
    
    # Load the rollout data
    data = load_rollout(rollout_file)
    
    if "error" in data:
        return jsonify({"error": data["error"]})
    
    # Generate HTML content
    html_content = generate_rollout_html(rollout_number, data)
    
    return jsonify({"html": html_content})

def generate_rollout_html(number, data):
    """Generate HTML content for a specific rollout."""
    # Format rewards
    rewards_text = format_rewards(
        data.get("reward_breakdown", {}),
        data.get("total_reward", 0)
    )
    
    # Get conversation parts and escape HTML tags
    prompt = data.get("prompt", "").replace("<", "&lt;").replace(">", "&gt;")
    response = data.get("response", "").replace("<", "&lt;").replace(">", "&gt;")
    
    return f"""
        <div class="rollout-info">
            <div class="rollout-title">Rollout {number}</div>
            <div class="rollout-rewards">{rewards_text}</div>
        </div>
        
        <div class="content-section">
            <div class="content-title">User (Prompt)</div>
            <div class="content-text">{prompt}</div>
        </div>
        
        <div class="content-section">
            <div class="content-title">Assistant (Response)</div>
            <div class="content-text">{response}</div>
        </div>
        
        <div class="stats">
            Answer: {data.get("answer", "N/A")} | 
            Answer Index: {data.get("answer_idx", "N/A")} | 
            Answer Text: {data.get("answer_text", "N/A")} | 
            Logprob Sum: {data.get("logprob_sum", "N/A"):.2f}
        </div>
    """

def main():
    """Start the web server."""
    rollouts = get_rollout_files()
    
    if not rollouts:
        print("No rollout files found.")
        return
    
    print(f"Found {len(rollouts)} rollout files.")
    print(f"Rollout range: {rollouts[0][0]} to {rollouts[-1][0]}")
    print("\nStarting web server...")
    print("Open your web browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server.")
    
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == "__main__":
    main()
