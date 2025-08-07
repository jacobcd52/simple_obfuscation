# Rollout Inspector

A web-based interface for inspecting rollout files generated during training.

## Usage

1. **Start the server:**
   ```bash
   python rollouts/inspect_rollouts.py
   ```

2. **Open your web browser and go to:**
   ```
   http://localhost:5001
   ```

3. **Navigate through rollouts:**
   - Enter a rollout number in the input field and click "Load"
   - Use "Previous" and "Next" buttons to navigate one by one
   - Use "+10" and "+100" buttons to skip ahead
   - Press Enter in the input field to load a specific rollout

## Features

- **Rollout Navigation:** Browse through all available rollout files
- **Reward Display:** View all reward components with 1 decimal place precision
- **Content Viewing:** See both the user prompt and assistant response as plain text
- **Statistics:** View answer details, indices, and logprob sums
- **Responsive Design:** Clean, modern interface that works on different screen sizes

## File Structure

- `inspect_rollouts.py` - The main script that runs the web server
- `rollout_*.json` - Individual rollout files (ignored by git)
- `README.md` - This documentation file

## Requirements

- Python 3.10+
- Flask 3.0+

The script will automatically install Flask if it's not already available.

## API Endpoints

- `GET /` - Main interface
- `GET /api/rollout/<number>` - Get rollout data as JSON

## Notes

- All JSON files in the rollouts folder are ignored by git (see `.gitignore`)
- The inspector script itself is tracked in git
- The web server runs on port 5001 by default
