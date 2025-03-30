# Slidding-puzzle-game
Project Overview
The Sliding Puzzle Game is a web-based implementation of the classic 3x3 sliding tile
puzzle. Users can upload their own images, which are automatically divided into tiles to
create an interactive puzzle. The game includes both manual solving capabilities and an AI-
powered automatic solver.
Technical Implementation
• Backend (Flask):
- Framework: Python Flask
- Key Libraries:
- PIL (Python Imaging Library) for image processing
- NumPy for numerical operations
- Werkzeug for file handling
- Heapq for priority queue implementation in A* algorithm
• Frontend
- Technologies:
- HTML5
- CSS3
- JavaScript (ES6+)
- Axios for HTTP requests
- Responsive Design: Adaptable layout using CSS Grid and Flexbox.
- Core Components:
• Image Processing
- Image upload with format validation
- Automatic tile generation (3×3 grid)
- Dynamic tile management with unique identifiers
- Support for multiple image formats (PNG, JPG, JPEG, GIF, BMP, WEBP)
• Puzzle Logic
- Solvability validation using inversion counting
- State management for puzzle configuration
- Valid move detection system
- Blank tile tracking
• AI Solver
- Implementation of A* pathfinding algorithm
- Manhattan distance heuristic
- State space exploration with priority queue
- Solution path reconstruction
Features
User Interface
- Clean, intuitive design
- Real-time move counter
- Visual feedback for tile movement
- Celebration animation on puzzle completion
- Error handling with user-friendly messages
Game Mechanics
- Image upload and automatic puzzle creation
- Manual puzzle solving through tile clicking
- AI-assisted solving with step-by-step animation
- Puzzle state validation
- Move tracking and statistics
- Technical Features
- Secure file handling
- Automatic puzzle solvability checking
- Efficient state management
- Asynchronous operations for smooth user experience
Implementation Details
• Puzzle Generation
1. Image upload and validation
2. Division into 3×3 grid
3. Random but solvable state generation
4. Tile path management
• AI Solver Algorithm
1. A* search implementation
2. Manhattan distance calculation
3. State space exploration
4. Solution path generation
5. Step-by-step move execution
State Management
1. Puzzle state tracking
2. Move validation
3. Solution verification
4. Tile position updates
Security Measures
- File type validation
- Secure filename handling
- Size limitations (16MB max)
- Server-side validation
- Error handling and sanitization
Performance Considerations
- Efficient image processing
- Optimized A* implementation
- Asynchronous operations
- Resource cleanup
- Memory management
Future Improvements
• Potential Enhancements
- Multiple difficulty levels
- Different grid sizes
- User accounts and scores
- Puzzle sharing capabilities
- Mobile optimization
• Technical Optimizations
- Solution caching
- Better heuristic functions
- Performance optimization
- Additional solver algorithms
Conclusion
The Sliding Puzzle Game successfully implements a classic puzzle with modern
features and AI capabilities. The combination of Flask backend and interactive
frontend provides a smooth user experience while maintaining good performance
and security standards.
