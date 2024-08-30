clear;clc;
%round(rand(5,2) * 5);

Psi_X1 = ...
    [4 0 7;
    9 8 4;
    8 9 7;
    10 7 2;
    7 8 7];


Psi_X1(:,3) = Psi_X1(:,1) + 1*Psi_X1(:,2);

%disp(det(Psi_X1' * Psi_X1))

Psi_X2 = ...
    [1 4;
    4 5;
    1 3;
    3 1;
    3 1];

Psi_X = [Psi_X1, Psi_X2];

K11 = [2 4 2;
    2 1 3;
    4 2 4];

Psi_Y1 = Psi_X1 * K11;
Psi_Y2= ...
    [2 5;
    2 1;
    4 4;
    3 4;
    3 2];

Psi_Y = [Psi_Y1, Psi_Y2];

[M, N] = size(Psi_X);
[~, N1] = size(Psi_X1);
[~, N2] = size(Psi_X2);

A = Psi_X1' * Psi_X1;
B = Psi_X1' * Psi_X2;
C = Psi_X2' * Psi_X1;
D = Psi_X2' * Psi_X2;
P = eye(M) - Psi_X2 * (Psi_X2' * Psi_X2)^(-1) * Psi_X2';
Q = Psi_X1' * P * Psi_X1;


K_approx = (Psi_X' * Psi_X)^(-1) * Psi_X' * Psi_Y;
K11_approx = K_approx(1:N1, 1:N1);

disp(K_approx)
disp(K11)

% K21에 block zero matrix는 여전하지만, K11이 큰 K에서 submatrix로 구했을 때는 달라져버림. 
