% Define the input matrix
A = rand(256);

% Initialize Q and R matrices
[m, n] = size(A);
Q = eye(m);
R = A;

% Preallocate cell arrays to store temporary Q and R matrices
tempQ = cell(1, n);
tempR = cell(1, n);

% Compute QR decomposition using parallel computing
parfor k = 1:n
    % Initialize temporary Q and R matrices
    tempQ{k} = Q;
    tempR{k} = R;
    
    % Compute the Householder reflector
    x = tempR{k}(k:m, k);
    e = zeros(length(x), 1);
    e(1) = 1;
    v = sign(x(1))*norm(x)*e + x;
    v = v / norm(v);
    
    % Update the Q matrix
    for j = 1:m
        tempQ{k}(j:m, k) = tempQ{k}(j:m, k) - 2*v(j:m)'*(v'*tempQ{k}(j:m, k));
    end
    
    % Update the R matrix
    tempR{k}(k:m, k:n) = tempR{k}(k:m, k:n) - 2*v*(v'*tempR{k}(k:m, k:n));
end

% Combine the temporary Q and R matrices
for k = 1:n
    Q = tempQ{k}*Q;
    R = tempR{k};
end
% Display results
disp("Q:");
disp(Q);
disp("R:");
disp(R);

