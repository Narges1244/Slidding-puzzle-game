import os
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import random
import numpy as np
from copy import deepcopy
import heapq

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PUZZLE_FOLDER = 'puzzles_tiles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PUZZLE_FOLDER'] = PUZZLE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PUZZLE_FOLDER, exist_ok=True)


class PuzzleNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost (g + h)
        
    def __lt__(self, other):
        return self.f < other.f
        
    def __eq__(self, other):
        return self.state == other.state

def get_blank_position(state):
    return state.index(0)

def get_manhattan_distance(state):
    """Calculate total Manhattan distance for all tiles"""
    distance = 0
    size = 3  # 3x3 puzzle
    for i in range(len(state)):
        if state[i] != 0:  # Skip blank tile
            current_row = i // size
            current_col = i % size
            # Calculate target position for the number
            target = state[i] - 1  # Subtract 1 because numbers are 1-8 (0 is blank)
            target_row = target // size
            target_col = target % size
            distance += abs(current_row - target_row) + abs(current_col - target_col)
    return distance

def get_valid_moves(blank_pos):
    """Get valid moves for blank tile"""
    moves = []
    row = blank_pos // 3
    col = blank_pos % 3
    
    # Check up
    if row > 0:
        moves.append(blank_pos - 3)
    # Check down
    if row < 2:
        moves.append(blank_pos + 3)
    # Check left
    if col > 0:
        moves.append(blank_pos - 1)
    # Check right
    if col < 2:
        moves.append(blank_pos + 1)
        
    return moves

def get_next_state(current_state, blank_pos, new_pos):
    """Generate new state after moving blank tile"""
    new_state = list(current_state)
    new_state[blank_pos], new_state[new_pos] = new_state[new_pos], new_state[blank_pos]
    return new_state


def solve_puzzle(initial_state):
    """
    Solve the puzzle using A* algorithm
    Returns: List of states representing the solution path
    """
    # Create start node
    start_node = PuzzleNode(initial_state)
    start_node.g = 0
    start_node.h = get_manhattan_distance(initial_state)
    start_node.f = start_node.g + start_node.h
    
    # Initialize open and closed lists
    closed_set = set()
    open_list = []
    heapq.heappush(open_list, (start_node.f, id(start_node), start_node))

    
    # Keep track of visited states to avoid cycles
    visited = {str(initial_state): start_node}
    
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # Updated goal state
    
    while open_list:
        current_node = heapq.heappop(open_list)[2]
        current_state_str = str(current_node.state)
            
        # Check if current state is goal state
        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return list(reversed(path))
        
        closed_set.add(current_state_str)
        blank_pos = current_node.state.index(0)
        
        for new_pos in get_valid_moves(blank_pos):
            new_state = get_next_state(current_node.state, blank_pos, new_pos)
            new_state_str = str(new_state)
            
            if new_state_str in closed_set:
                continue
                
            new_node = PuzzleNode(new_state, current_node)
            new_node.g = current_node.g + 1
            new_node.h = get_manhattan_distance(new_state)
            new_node.f = new_node.g + new_node.h
            
            if new_state_str not in visited or visited[new_state_str].f > new_node.f:
                heapq.heappush(open_list, (new_node.f, id(new_node), new_node))
                visited[new_state_str] = new_node
    
    return None  # No solution found
        
def get_inversion_count(arr):
    """
    Calculate the number of inversions in the puzzle state.
    Inversions are pairs of tiles where a larger number comes before a smaller number.
    """
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            # Don't count blank tile (0) in inversions
            if arr[i] != 0 and arr[j] != 0 and arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def is_solvable(puzzle_state, blank_position):
    """
    Check if a puzzle configuration is solvable.
    For 3x3 puzzles:
    - If blank is on even row from bottom: solvable if inversion count is odd
    - If blank is on odd row from bottom: solvable if inversion count is even
    """
    inv_count = get_inversion_count(puzzle_state)
    blank_row = blank_position // 3
    from_bottom = 2 - blank_row  # For 3x3 puzzle
    
    if from_bottom % 2 == 0:
        return inv_count % 2 == 1
    else:
        return inv_count % 2 == 0
    
def generate_solvable_puzzle():
    """
    Generate a solvable puzzle configuration.
    Returns the tile arrangement and blank position.
    """
    while True:
        # Create list of numbers 0-8 (0 represents blank)
        puzzle_state = list(range(9))
        # Shuffle the list
        random.shuffle(puzzle_state)
        
        # Get blank position
        blank_pos = puzzle_state.index(0)
        
        # Check if configuration is solvable
        if is_solvable(puzzle_state, blank_pos):
            print("Puzzle is solvable.")
            return puzzle_state, blank_pos
        else:
            print("Puzzle is not solvable. Regenerating...")

def create_puzzle_tiles(image_path):
    """
    Divide the image into tiles and create a solvable puzzle arrangement
    """
    # Clean up the puzzle folder first
    for filename in os.listdir(app.config['PUZZLE_FOLDER']):
        file_path = os.path.join(app.config['PUZZLE_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')
    
    
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        tile_width = width // 3
        tile_height = height // 3
        
        # Create all tiles first
        tiles = []
        for i in range(9):
            row = i//3
            col = i % 3

            left = col * tile_width
            top = row * tile_height
            right = left + tile_width
            bottom = top + tile_height
                
            tile = img.crop((left, top, right, bottom))
            tiles.append(tile)
        
        # Generate solvable puzzle arrangement
        puzzle_state, blank_index = generate_solvable_puzzle()
        
        # Create tiles in shuffled order
        tile_paths = []
        for i in range(9):
            if puzzle_state[i] == 0:  # Blank tile
                
                blank_tile = Image.new('RGB', (tile_width, tile_height), color='gray')
                tile_filename = f'blank_{random.randint(1000, 9999)}.png'
                tile_path = os.path.join(app.config['PUZZLE_FOLDER'], tile_filename)
                blank_tile.save(tile_path)
                tile_paths.append(f'/puzzles_tiles/{tile_filename}')
            else:
                # Save the actual tile
                original_index = puzzle_state[i]
                tile_filename = f'tile_{i}_{original_index}_{random.randint(1000, 9999)}.png'
                tile_path = os.path.join(app.config['PUZZLE_FOLDER'], tile_filename)
                tiles[original_index].save(tile_path)
                tile_paths.append(f'/puzzles_tiles/{tile_filename}')
        
        return {
            'tile_paths': tile_paths,
            'blank_index': blank_index,
            'puzzle_state': puzzle_state,
            'dimensions': {
                'tile_width': tile_width,
                'tile_height': tile_height,
                'original_width': width,
                'original_height': height
            }
        }        


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_puzzle_tile(image_path):
    """
    Divide the image into 9 equal tiles and save them
    Returns a list of tile paths and the blank tile index
    """
    # Clean up the puzzle folder
    for filename in os.listdir(app.config['PUZZLE_FOLDER']):
        file_path = os.path.join(app.config['PUZZLE_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')        
    # Open the image
    with Image.open(image_path) as img:
        #Ensure the image is a RGB mode
        img = img.convert('RGB')

        #Get original image dimensions
        width, height = img.size

        #Calculate tile dimensions
        tile_width = width // 3
        tile_height = height // 3

        # List to store tile path
        tile_paths = []

        #Generate 9 tiles
        for row in range(3):
            for col in range(3):
                # Check if this is the bottom right tile
                if row == 2 and col == 2:
                    # Create a black tile for bottom right
                    tile_filename = 'tile_blank.png'
                    tile_path = os.path.join(app.config['PUZZLE_FOLDER'], tile_filename)
                    # Create a black tile
                    black_tile = Image.new('RGB', (tile_width, tile_height), color='black')
                    black_tile.save(tile_path)
                else:

                    left = col * tile_width
                    top = row * tile_height
                    right = left + tile_width
                    bottom = top + tile_height

                    tile = img.crop((left, top, right, bottom))
                    tile_filename = f'tile_{row}_{col}.png'
                    tile_path = os.path.join(app.config['PUZZLE_FOLDER'], tile_filename)
                    tile.save(tile_path)
                
                tile_paths.append(f'/puzzles_tiles/{tile_filename}')

        
        
        return {
            'tile_paths': tile_paths,
            'blank_index': 8,
            'puzzle_state': list(range(9)),
            'dimensions': {
                'tile_width': tile_width,
                'tile_height': tile_height,
                'original_width': width,
                'original_height': height
            }
        }

@app.route('/')
def upload_form():
    """Render the upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing."""
    # Check if file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(filepath)
        
       
        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename
         }), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/shuffle', methods=['POST'])
def shuffle_puzzle():
    """Create shuffled puzzle from uploaded image"""
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
        
        # Create puzzle tiles with solvable configuration
        puzzle_info = create_puzzle_tiles(filepath)
        
        return jsonify({
            'message': 'Puzzle created successfully',
            'tile_paths': puzzle_info['tile_paths'],
            'blank_index': puzzle_info['blank_index'],
            'puzzle_state': puzzle_info['puzzle_state'],
            'dimensions': puzzle_info['dimensions']
        }), 200
    except Exception as e:
        print(f"Error creating puzzle: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/puzzles_tiles/<filename>')
def serve_tile(filename):
    """Serve puzzle tiles"""
    return send_file(os.path.join(app.config['PUZZLE_FOLDER'], filename))
@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        if not data :
            return jsonify({'error': 'No data provided'}), 400
        if 'puzzleState' not in data:
            return jsonify({'error': 'No puzzle state provided'}), 400
            
        initial_state = data['puzzleState']
        print("intial state:",initial_state)

        # Validate puzzle state
        if not isinstance(initial_state, list) or len(initial_state) != 9:
            return jsonify({'error': 'Invalid puzzle state format'}), 400
            
            
        try:
            initial_state = [int(x) for x in initial_state]
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid number in puzzle state: {str(e)}'}), 400

        # Validate numbers
        if not all(0 <= x <= 8 for x in initial_state):
            return jsonify({'error': 'All numbers must be between 0 and 8'}), 400
        if len(set(initial_state)) != 9:
            return jsonify({'error': 'All numbers must be unique'}), 400
        
        # Check if puzzle is solvable
        blank_pos = initial_state.index(0)
        if not is_solvable(initial_state, blank_pos):
            return jsonify({'error': 'Puzzle configuration is not solvable'}), 400
            
        print(f"Puzzle state validated. Blank position: {blank_pos}")  # Debug log
        
        # Get solution
        solution_path = solve_puzzle(initial_state)
        
        if solution_path is None:
            return jsonify({
                'error': 'No solution found for this configuration'
            }), 400
            
        print(f"Solution path found with {len(solution_path)} steps")
        for i, state in enumerate(solution_path):
            print(f"Step {i}: {state}")
       
       
         # Create solution moves
        solution_moves = []
        current_state = initial_state.copy()
        current_tiles = data.get('tilePaths', [])
        
        # Add initial position
        solution_moves.append({
            'state': current_state.copy(),
            'tiles': current_tiles.copy()
        })
        
        # Generate moves
        for next_state in solution_path[1:]:
            old_blank = current_state.index(0)
            new_blank = next_state.index(0)
            
            if current_tiles:
                current_tiles[old_blank], current_tiles[new_blank] = \
                    current_tiles[new_blank], current_tiles[old_blank]
            current_state = next_state.copy()
            
            solution_moves.append({
                'state': current_state.copy(),
                'tiles': current_tiles.copy() if current_tiles else []
            })
            
        return jsonify({
            'success': True,
            'solution': solution_moves,
            'steps': len(solution_moves) - 1
        }), 200
        
    except Exception as e:
        print(f"Error in solve endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)