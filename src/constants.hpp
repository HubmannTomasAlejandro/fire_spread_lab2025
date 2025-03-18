#ifndef CONSTANTS_H
#define CONSTANTS_H

// Mathematical constants
constexpr double EULER = 2.718281828459045;

// Simulation parameters
constexpr int MAX_ITERATIONS = 1000;
constexpr double FIRE_SPREAD_PROB = 0.3;

// Wind direction angles (in radians)
constexpr double UP_LEFT_ANGLE = 3 * M_PI / 4;
constexpr double UP_ANGLE = M_PI;
constexpr double UP_RIGHT_ANGLE = 5 * M_PI / 4;
constexpr double LEFT_ANGLE = M_PI / 2;
constexpr double RIGHT_ANGLE = 3 * M_PI / 2;
constexpr double DOWN_LEFT_ANGLE = M_PI / 4;
constexpr double DOWN_ANGLE = 0;
constexpr double DOWN_RIGHT_ANGLE = 7 * M_PI / 4;
constexpr double ANGLES[8] = {
    UP_LEFT_ANGLE, UP_ANGLE, UP_RIGHT_ANGLE, LEFT_ANGLE,
    RIGHT_ANGLE, DOWN_LEFT_ANGLE, DOWN_ANGLE, DOWN_RIGHT_ANGLE
};

// Cell moves
constexpr int UP_LEFT[2] = { -1, -1 };
constexpr int UP[2] = { -1, 0 };
constexpr int UP_RIGHT[2] = { -1, 1 };
constexpr int LEFT[2] = { 0, -1 };
constexpr int RIGHT[2] = { 0, 1 };
constexpr int DOWN_LEFT[2] = { 1, -1 };
constexpr int DOWN[2] = { 1, 0 };
constexpr int DOWN_RIGHT[2] = { 1, 1 };

constexpr int MOVES[8][2] = {
    {UP_LEFT[0], UP_LEFT[1]}, {UP[0], UP[1]}, {UP_RIGHT[0], UP_RIGHT[1]}, {LEFT[0], LEFT[1]},
    {RIGHT[0], RIGHT[1]}, {DOWN_LEFT[0], DOWN_LEFT[1]}, {DOWN[0], DOWN[1]}, {DOWN_RIGHT[0], DOWN_RIGHT[1]}
};

#endif // CONSTANTS_H