#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace cv;
using namespace std;

struct DetectedShape {
    char type;
    Point center;
    int grid_pos = -1;
};

struct MinimaxResult {
    int score;
    int row;
    int col;
};

char checkWinner(const vector<vector<char>>& board) {
    for (int i = 0; i < 3; i++) {
        if (board[i][0] != ' ' && board[i][0] == board[i][1] && board[i][1] == board[i][2]) {
            return board[i][0];
        }
    }

    for (int i = 0; i < 3; i++) {
        if (board[0][i] != ' ' && board[0][i] == board[1][i] && board[1][i] == board[2][i]) {
            return board[0][i];
        }
    }

    if (board[0][0] != ' ' && board[0][0] == board[1][1] && board[1][1] == board[2][2]) {
        return board[0][0];
    }
    if (board[0][2] != ' ' && board[0][2] == board[1][1] && board[1][1] == board[2][0]) {
        return board[0][2];
    }

    return ' ';
}

bool isBoardFull(const vector<vector<char>>& board) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == ' ') return false;
        }
    }
    return true;
}

pair<int, int> countSymbols(const vector<vector<char>>& board) {
    int x_count = 0, o_count = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 'X') x_count++;
            else if (board[i][j] == 'O') o_count++;
        }
    }
    return { x_count, o_count };
}

bool isGameValid(const vector<vector<char>>& board) {
    auto [x_count, o_count] = countSymbols(board);

    if (abs(x_count - o_count) > 1) return false;

    char winner = checkWinner(board);

    if (winner != ' ') {
        if (winner == 'X' && x_count < o_count) return false;
        if (winner == 'O' && o_count > x_count) return false;

        vector<vector<char>> temp_board = board;
        char other_player = (winner == 'X') ? 'O' : 'X';

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (temp_board[i][j] == winner) temp_board[i][j] = ' ';
            }
        }

        if (checkWinner(board) != winner && checkWinner(temp_board) == other_player) {
            return false;
        }
    }

    return true;
}

MinimaxResult minimax(vector<vector<char>>& board, int depth, bool isMaximizing, char aiPlayer, char humanPlayer) {
    char winner = checkWinner(board);

    if (winner == aiPlayer) return { 10 - depth, -1, -1 };
    if (winner == humanPlayer) return { depth - 10, -1, -1 };
    if (isBoardFull(board)) return { 0, -1, -1 };

    if (isMaximizing) {
        MinimaxResult best = { numeric_limits<int>::min(), -1, -1 };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] == ' ') {
                    board[i][j] = aiPlayer;
                    MinimaxResult result = minimax(board, depth + 1, false, aiPlayer, humanPlayer);
                    board[i][j] = ' ';

                    if (result.score > best.score) {
                        best = { result.score, i, j };
                    }
                }
            }
        }
        return best;
    }
    else {
        MinimaxResult best = { numeric_limits<int>::max(), -1, -1 };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] == ' ') {
                    board[i][j] = humanPlayer;
                    MinimaxResult result = minimax(board, depth + 1, true, aiPlayer, humanPlayer);
                    board[i][j] = ' ';

                    if (result.score < best.score) {
                        best = { result.score, i, j };
                    }
                }
            }
        }
        return best;
    }
}

void analyzeGame(const vector<vector<char>>& board) {
    cout << "\n=== GAME ANALYSIS ===" << endl;

    if (!isGameValid(board)) {
        cout << "IMPOSSIBLE GAME! Invalid move sequence." << endl;
        return;
    }

    char winner = checkWinner(board);
    if (winner != ' ') {
        cout << "WINNER: " << winner << endl;
        return;
    }

    if (isBoardFull(board)) {
        cout << "DRAW! Board is full." << endl;
        return;
    }

    auto [x_count, o_count] = countSymbols(board);
    char current_player = (x_count == o_count) ? 'X' : 'O';
    char opponent = (current_player == 'X') ? 'O' : 'X';

    cout << "Game in progress. Current player: " << current_player << endl;

    vector<vector<char>> board_copy = board;
    MinimaxResult best_move = minimax(board_copy, 0, true, current_player, opponent);

    if (best_move.row != -1 && best_move.col != -1) {
        cout << "OPTIMAL MOVE: row " << best_move.row + 1
            << ", column " << best_move.col + 1
            << " (position [" << best_move.row << "][" << best_move.col << "])" << endl;

        if (best_move.score > 0) {
            cout << "Evaluation: Player " << current_player << " can win!" << endl;
        }
        else if (best_move.score < 0) {
            cout << "Evaluation: Player " << opponent << " has the advantage." << endl;
        }
        else {
            cout << "Evaluation: Perfect play leads to a draw." << endl;
        }
    }
}

DetectedShape analyzeContour(const vector<Point>& contour, const Vec4i& hierarchyEntry, const vector<vector<Point>>& allContours) {
    double area = contourArea(contour);
    char detected_type = ' ';
    int parent_idx = hierarchyEntry[3];
    if (parent_idx != -1) {
        double parent_area = contourArea(allContours[parent_idx]);
        if (parent_area > 0 && area / parent_area > 0.3) {
            return { detected_type, Point() };
        }
    }
    if (area < 500 || area > 8000) {
        return { detected_type, Point() };
    }

    double perimeter = arcLength(contour, true);
    Rect bbox = boundingRect(contour);
    double roundness = area / perimeter;

    Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    if (roundness > 10) {
        detected_type = 'O';
    }
    else {
        detected_type = 'X';
    }

    return { detected_type, center };
}

Rect findGameBoard(const Mat& img_thresh) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img_thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double max_area = 0;
    Rect board_bbox;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > max_area) {
            max_area = area;
            board_bbox = boundingRect(contour);
        }
    }
    if (max_area < 5000) {
        return Rect();
    }
    return board_bbox;
}

vector<vector<char>> createGameBoard(const vector<DetectedShape>& shapes, const Rect& gridBBox) {
    vector<vector<char>> board(3, vector<char>(3, ' '));
    int cell_width = gridBBox.width / 3;
    int cell_height = gridBBox.height / 3;

    for (const auto& shape : shapes) {
        int rel_x = shape.center.x - gridBBox.x;
        int rel_y = shape.center.y - gridBBox.y;
        int row = clamp(rel_y / cell_height, 0, 2);
        int col = clamp(rel_x / cell_width, 0, 2);
        board[row][col] = shape.type;
    }
    return board;
}

void printGameBoard(const vector<vector<char>>& board) {
    cout << "\n=== CURRENT BOARD ===" << endl;
    for (int r = 0; r < 3; ++r) {
        cout << "---|---|---" << endl;
        cout << " " << board[r][0] << " | " << board[r][1] << " | " << board[r][2] << endl;
    }
    cout << "---|---|---" << endl;
}

int main() {
    for (int i = 0; i <= 9; ++i) {
        string filename = "Resources/tic-tac-toe/test_" + to_string(i) + ".jpg";
        Mat img = imread(filename);

        if (img.empty()) {
            cout << "Failed to load " << filename << endl;
            continue;
        }

        cout << "\n\n========================================" << endl;
        cout << "PROCESSING IMAGE: " << filename << endl;
        cout << "========================================" << endl;

        resize(img, img, Size(), 0.4, 0.4);
        medianBlur(img, img, 5);

        Mat img_display = img.clone();
        Mat img_gray, img_thresh;
        cvtColor(img, img_gray, COLOR_BGR2GRAY);
        Mat img_thresh_shapes;
        adaptiveThreshold(img_gray, img_thresh_shapes, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        dilate(img_thresh_shapes, img_thresh_shapes, kernel);
        Mat img_thresh_board;
        adaptiveThreshold(img_gray, img_thresh_board, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
        Rect gridBBox = findGameBoard(img_thresh_board);

        if (!gridBBox.empty()) {
            rectangle(img_display, gridBBox, Scalar(0, 255, 255), 2);
        }

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(img_thresh_shapes, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        vector<DetectedShape> detectedShapes;
        for (size_t j = 0; j < contours.size(); j++) {
            DetectedShape shape = analyzeContour(contours[j], hierarchy[j], contours);
            if (shape.type != ' ') {
                detectedShapes.push_back(shape);
                Scalar color = (shape.type == 'O') ? Scalar(0, 255, 0) : Scalar(255, 0, 0);
                putText(img_display, string(1, shape.type), shape.center, FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
        }

        vector<vector<char>> board = createGameBoard(detectedShapes, gridBBox);
        printGameBoard(board);
        analyzeGame(board);

        string window_name_result = "Detected shapes (" + to_string(i) + ")";
        imshow(window_name_result, img_display);

        waitKey(0);
    }
    destroyAllWindows();
    return 0;
}