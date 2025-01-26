function [Q, R] = CGS(A)
    n = length(A);
    Q = A;
    R = zeros(n);

    for k = 1 : n
        v = A(:,k); % v is the kth column in A
        for j = 1 : k -1
            q = Q(:,j);
            R(j, k) = q' * A(:,k);
            v = v - R(j, k) * q;
        end
        R(k,k) = norm(v);
        Q(:,k) = v/R(k,k);
    end
end

function [Q, R] = MGS(A)
    n = length(A);
    Q = A;
    R = zeros(n);

    for k = 1 : n
        R(k,k) = norm(Q(:,k));
        Q(:,k) = Q(:,k)/R(k,k);

        for j = k+1 : n
            R(k,j) = Q(:,k)' * Q(:,j);
            Q(:,j) = Q(:,j) - R(k,j) * Q(:,k);
        end 
    end
end

% Define the MSE function
function mse = MSE(A, A_, n)
  sum = 0;
  for i = 1 : n
    for j = 1 : n
      sum = sum + single((A_(i,j) - A(i,j))^2); % Convert difference to single before squaring
    end
  end
  mse = single(sum/n); % Convert sum and final result to single
end

% MGS function (basic implementation)
function [Q, R] = MGS(A)
  [m, n] = size(A);
  Q = single(zeros(m, n)); % Allocate memory for single-precision Q
  R = single(zeros(n, n)); % Allocate memory for single-precision R
  for i = 1:n
    v = single(A(:, i)); % Convert column vector to single
    for j = 1:i-1
      R(j, i) = Q(:, j)' * v;
      v = v - single(R(j, i) * Q(:, j)); % Convert product to single before subtraction
    end
    R(i, i) = norm(v);
    Q(:, i) = v / R(i, i);
  end
end

% Main Script
n = 1000; % Increase number of simulations for better results
k = 32; % matrix size
mse_cgs = zeros(n, 1);
mse_mgs = zeros(n, 1);

% Loop for simulations
for i = 1 : n
  A = single(10 * rand(k, k)); % Generate single-precision random matrix
  [q, r] = CGS(A); % Assuming CGS function is defined elsewhere (single-precision?)
  [Q, R] = MGS(A);
  mse_cgs(i) = MSE(A, q*r, k);
  mse_mgs(i) = MSE(A, Q*R, k);
end

% Calculate average MSE
mean_cgs = mean(mse_cgs);
mean_mgs = mean(mse(mse_mgs));

% Plotting individual errors (replace with your preferred plotting method)
figure;
plot(1:n, mse_cgs, 'b-', 1:n, mse_mgs, 'r--');
xlabel('Iteration (i)');
ylabel('Mean Squared Error');
title('Mean Squared Error Comparison between CGS and MGS (single)');
legend('CGS', 'MGS');

% Display results
disp("Average MSE for CGS (single):");
disp(mean_cgs);
disp("Average MSE for MGS (single):");
disp(mean_mgs);

% Note: Analyze individual errors as before
