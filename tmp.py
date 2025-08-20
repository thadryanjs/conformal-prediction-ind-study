""" "
You're asking for a concrete implementation of Optimal Model Design (OMD) for the tabular case using only NumPy, focusing on the update functions for the inner and outer loops as described in Algorithm 1. You also mentioned that you're less familiar with the implementation details of minimization methods.
In the tabular setting, the core elements of OMD are realized through specific numerical operations on arrays (tables) representing states, actions, rewards, and transitions.
Key Concepts for Tabular OMD Implementation
1. Tabular Representation:
    ◦ Q-function (Q): A NumPy array of shape (S, A), where S is the number of states and A is the number of actions. This directly corresponds to the w in the Implicit Function Theorem (IFT) context.
    ◦ Reward Function (r): A NumPy array of shape (S, A).
    ◦ Transition Probabilities (p): A NumPy array of shape (S, A, S), representing p(s'|s, a).
    ◦ Model Parameters (θ): In the tabular case, θ effectively comprises the learned reward function (r_theta) and learned transition probabilities (p_theta). The source uses logits for p_theta to ensure they are differentiable and can be converted to probabilities via softmax.
2. Inner Loop: Finding Q* (The Implicit Function ϕ(θ)):
    ◦ For a given set of model parameters θ = (r_theta, p_theta_logits), the goal is to find the optimal Q-function Q* that satisfies the Bellman optimality equation induced by the model: Q(s, a) = BθQ(s, a) for all state-action pairs.
    ◦ In tabular MDPs, this Q* is found by Value Iteration until convergence [previous response, 19]. This contrasts with the function approximation case (Section 6), where w (Q-network parameters) are updated for K steps to minimize Bellman error. For tabular, we seek exact convergence to the fixed point.
    ◦ The soft Bellman optimality operator Bθ is defined as BθQ(s, a) = rθ(s, a) + γEpθ(s′|s,a) log ∑ a′ expQ(s′, a′). The logsumexp term is chosen for differentiability.
3. Outer Loop: Updating Model Parameters θ:
    ◦ The objective is to train θ to either maximize expected returns J(πQ) or minimize the Bellman error Ltrue(Q) induced by the true MDP. The sources indicate Ltrue can be more stable for optimization. For this implementation, we will use Ltrue(Q) as the outer objective, which means we perform gradient descent.
    ◦ The gradient ∂Ltrue(Q*)/∂θ is computed using the Implicit Function Theorem (IFT).
    ◦ The general form of the IFT for ∂ϕ(θ)/∂θ (which is ∂Q*/∂θ in our context) is − ( ∂f(θ, Q*)/∂Q )⁻¹ ⋅ ∂f(θ, Q*)/∂θ.
        ▪ Here, f(θ, Q) = Q - BθQ = 0 represents the Bellman constraint [previous response].
    ◦ The full gradient for the outer objective becomes: ∂Ltrue(Q*)/∂θ = (∂Ltrue/∂Q*) ⋅ (∂Q*/∂θ) (using the chain rule) [similar to 35]. This involves:
        ▪ ∂Ltrue/∂Q*: The gradient of the outer objective (Ltrue) with respect to Q*.
        ▪ ∂f/∂Q: The Jacobian of the constraint function f with respect to Q.
        ▪ ∂f/∂θ: The Jacobian of the constraint function f with respect to the model parameters θ.
    ◦ These Jacobians are full matrices in the tabular case, involving all state-action pairs.
Imperative Python Code (NumPy Only)
Below is the structured code for a tabular MDP, including detailed comments mapping to the theoretical concepts and source material.
"""

import numpy as np

# --- 1. Define MDP Parameters and Initialize Model Parameters ---

# Global MDP settings (for simplicity in this example)
S = 2  # Number of states
A = 2  # Number of actions
gamma = 0.9  # Discount factor (gamma ∈ [0, 1)) [10]
alpha_softmax = 0.01  # Temperature for softmax in policy (pi_Q) [7, 11]

# True MDP definition (for Ltrue objective and comparison)
# r(s,a) is reward for state-action pair (s,a)
r_true = np.array(
    [[1.0, 0.0], [0.0, 1.0]]  # r(s=0, a=0), r(s=0, a=1)  # r(s=1, a=0), r(s=1, a=1)
)
# p(s'|s,a) is transition probability from (s,a) to s'
p_true = np.array(
    [  # Shape (S, A, S)
        [[0.9, 0.1], [0.1, 0.9]],  # s=0, a=0 -> s'=0, s'=1  # s=0, a=1 -> s'=0, s'=1
        [[0.8, 0.2], [0.2, 0.8]],  # s=1, a=0 -> s'=0, s'=1  # s=1, a=1 -> s'=0, s'=1
    ]
)
rho0 = np.array([0.5, 0.5])  # Initial state distribution [10]

# Model parameters (theta) - These are what OMD learns
# Initialized randomly for the example
r_theta = np.random.rand(S, A)  # Learned reward function (r_theta(s,a))
p_theta_logits = np.random.rand(
    S, A, S
)  # Learned transition logits (will be softmaxed to get p_theta(s'|s,a)) [4]

# --- 2. Helper Functions for Mathematical Operations ---


def logsumexp(arr, axis=None):
    """
    Numerically stable log-sum-exp implementation using NumPy only.
    Used in the soft Bellman operator. [7]
    """
    max_val = np.max(arr, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=True)) + max_val


def softmax_q(Q_s_prime_row, alpha):
    """
    Computes softmax probabilities from Q-values for a given state s' row (all actions).
    Used for the policy pi_Q(a|s) = exp(Q(s,a)/alpha) / sum_a' exp(Q(s,a')/alpha). [7]
    """
    # Scale Q-values by alpha before exponentiation for temperature effect
    exp_Q_scaled = np.exp(Q_s_prime_row / alpha)
    return exp_Q_scaled / np.sum(exp_Q_scaled)


def softmax_p_logits(logits, axis=-1):
    """
    Converts model's transition logits into probabilities using softmax. [4]
    """
    max_val = np.max(logits, axis=axis, keepdims=True)
    exp_logits_scaled = np.exp(logits - max_val)
    return exp_logits_scaled / np.sum(exp_logits_scaled, axis=axis, keepdims=True)


# --- 3. Inner Loop: find_Q_star_tabular ---


def find_Q_star_tabular(
    r_model,
    p_model_logits,
    gamma,
    alpha_softmax,
    Q_init=None,
    tol=1e-6,
    max_iterations=1000,
):
    """
    **Inner Loop Update Function for Tabular Case.**

    Finds the Q-function (Q*) that is a fixed point of the soft Bellman optimality operator
    induced by the given model (r_model, p_model_logits).

    This is achieved by performing **Value Iteration until convergence**. [previous response, 19]
    In Algorithm 1, this corresponds to the "Update Qw parameters w to minimize L(theta, w)" step [1],
    but for tabular, 'w' *is* the Q-table, and 'minimizing L' means finding the fixed point.

    Args:
        r_model (np.array): Current learned reward function.
        p_model_logits (np.array): Current learned transition logits.
        gamma (float): Discount factor.
        alpha_softmax (float): Temperature for the log-sum-exp in the soft Bellman.
        Q_init (np.array, optional): Initial Q-table for warm-starting value iteration.
        tol (float): Tolerance for convergence.
        max_iterations (int): Maximum iterations for value iteration.

    Returns:
        np.array: The converged Q-table (Q* or w*).
    """
    # Initialize Q-table
    if Q_init is None:
        Q = np.zeros((S, A))
    else:
        Q = np.copy(Q_init)

    p_model = softmax_p_logits(
        p_model_logits
    )  # Convert logits to probabilities for calculation

    for iteration in range(max_iterations):
        Q_prev = Q.copy()
        for s in range(S):
            for a in range(A):
                # Calculate B_theta Q(s,a) as per Eq. (3) [7]
                reward_term = r_model[s, a]

                # Term: gamma * Ep_theta(s'|s,a) [log sum_a' exp(Q(s', a'))]
                # Note: Eq. (3) shows 'log sum_a' expQ', implying alpha=1 for this term.
                # If a different alpha is desired here, it needs to be applied to Q[s_prime, :].
                next_state_values_lse = np.array(
                    [logsumexp(Q[s_prime, :]).item() for s_prime in range(S)]
                )

                transition_term = np.sum(p_model[s, a, :] * next_state_values_lse)

                Q[s, a] = reward_term + gamma * transition_term

        # Check for convergence
        if np.max(np.abs(Q - Q_prev)) < tol:
            # print(f"  Value iteration converged in {iteration + 1} iterations.")
            break
    else:
        print("Warning: Value iteration did not converge within max_iterations.")
    return Q


# --- 4. Outer Loop: update_model_parameters_tabular ---


def compute_Ltrue(Q_star_model, r_true_mdp, p_true_mdp, gamma_mdp):
    """
    Computes the Bellman error (Ltrue) as the outer loop objective. [8, 12]
    Ltrue(Q) = sum_s,a (Q(s,a) - BQ(s,a))^2, where B is induced by the TRUE MDP (r_true, p_true).
    """
    bellman_errors_sq = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            # Calculate BQ(s,a) induced by the true model (r_true, p_true)
            reward_term_true = r_true_mdp[s, a]

            # log sum_a' exp(Q(s', a'))
            next_state_values_lse_true = np.array(
                [logsumexp(Q_star_model[s_prime, :]).item() for s_prime in range(S)]
            )

            transition_term_true = np.sum(
                p_true_mdp[s, a, :] * next_state_values_lse_true
            )

            BQ_sa = reward_term_true + gamma_mdp * transition_term_true

            bellman_errors_sq[s, a] = (Q_star_model[s, a] - BQ_sa) ** 2

    return np.sum(bellman_errors_sq)


# --- Jacobian building functions for IFT ---


def compute_jacobian_f_Q(Q_val, r_model, p_model_logits, gamma):
    """
    Computes the Jacobian matrix ∂f/∂Q, where f(Q, θ) = Q - B_theta Q.
    This is the (∂f(θ, w*)/∂w) term from the IFT formula. [3]
    Shape: (S*A) x (S*A)
    """
    p_model = softmax_p_logits(p_model_logits)  # p_theta(s'|s,a)
    SA = S * A  # Total number of Q-values
    J_f_Q = np.eye(
        SA
    )  # Initialize as Identity matrix (for the Q term in f = Q - B_theta Q)

    for i_sa in range(SA):  # Iterate over output (s,a) of f
        s, a = np.unravel_index(i_sa, (S, A))

        for j_s_prime_a_prime in range(SA):  # Iterate over input (s',a') from Q
            s_prime, a_prime = np.unravel_index(j_s_prime_a_prime, (S, A))

            # The derivative ∂B_theta Q(s,a) / ∂Q(s_prime, a_prime)
            # This term is non-zero only if 's_prime' is a possible next state from (s,a)
            # and 'a_prime' is an action at 's_prime' whose Q-value influences the logsumexp.

            # The derivative of logsumexp(Q(s_prime,:)) with respect to Q(s_prime, a_prime) is softmax(Q(s_prime,:))[a_prime]
            # Assuming alpha=1 in logsumexp (as per Eq. 3)
            softmax_term_for_deriv = np.exp(Q_val[s_prime, a_prime]) / np.sum(
                np.exp(Q_val[s_prime, :])
            )

            derivative_part = gamma * p_model[s, a, s_prime] * softmax_term_for_deriv

            # Subtract this from the identity matrix part because f = Q - B_theta Q
            J_f_Q[i_sa, j_s_prime_a_prime] -= derivative_part
    return J_f_Q


def compute_jacobian_f_theta(Q_val, r_model, p_model_logits, gamma):
    """
    Computes the Jacobian matrix ∂f/∂θ, where f(Q, θ) = Q - B_theta Q.
    This is the (∂f(θ, w*)/∂θ) term from the IFT formula. [3]
    θ comprises (r_theta, p_theta_logits).
    Shape: (S*A) x (S*A + S*A*S) -- (number of Q-values) x (total number of model parameters)
    """
    p_model = softmax_p_logits(p_model_logits)  # Current p_theta(s'|s,a)
    SA = S * A

    num_r_params = S * A
    num_p_logits_params = S * A * S
    total_theta_params = num_r_params + num_p_logits_params

    J_f_theta = np.zeros((SA, total_theta_params))

    # Calculate logsumexp values for all states s' (these are independent of model params)
    lse_values_per_s_prime = np.array(
        [logsumexp(Q_val[s_prime, :]).item() for s_prime in range(S)]
    )

    # Part 1: Derivatives with respect to r_model parameters
    # ∂f_sa / ∂r_model(s',a') = -∂r_model(s,a)/∂r_model(s',a')
    # This is -1 if (s,a) == (s',a') and 0 otherwise.
    for i_sa in range(SA):
        r_param_idx = i_sa  # flat index for r_model(s,a) parameter
        J_f_theta[i_sa, r_param_idx] = -1.0

    # Part 2: Derivatives with respect to p_model_logits parameters
    # ∂f_sa / ∂p_model_logits(s_target, a_target, s_prime_k)
    # The derivative is non-zero only if (s,a) from f_sa matches (s_target, a_target)
    for i_sa in range(SA):  # (s,a) for the output f_sa
        s, a = np.unravel_index(i_sa, (S, A))

        for s_prime_k in range(
            S
        ):  # Index for the specific s' in p_model_logits(s,a,s')
            p_logits_param_idx = num_r_params + np.ravel_multi_index(
                (s, a, s_prime_k), (S, A, S)
            )

            # Derivative of the sum_s_hat p_theta(s_hat|s,a) * LSE(s_hat) term
            # with respect to p_theta_logits(s,a,s_prime_k)
            # This is sum_s_hat [ (p_theta(s_hat|s,a) * (delta_s_hat,s_prime_k - p_theta(s_prime_k|s,a))) * LSE(s_hat) ]
            sum_over_s_hat = 0.0
            for s_hat in range(S):
                # Derivative of p_model(s_hat | s, a) with respect to p_model_logits(s,a,s_prime_k)
                # This is the derivative of softmax: p_j * (delta_jk - p_k)
                p_derivative_term = p_model[s, a, s_hat] * (
                    1.0 if s_hat == s_prime_k else 0.0
                )
                p_derivative_term -= p_model[s, a, s_hat] * p_model[s, a, s_prime_k]

                sum_over_s_hat += p_derivative_term * lse_values_per_s_prime[s_hat]

            J_f_theta[i_sa, p_logits_param_idx] = -gamma * sum_over_s_hat

    return J_f_theta


def compute_grad_Ltrue_Q(Q_star_model, true_r, true_p, gamma):
    """
    Computes the gradient of the outer objective Ltrue(Q*) with respect to Q*.
    This is the ∂Ltrue/∂Q* term (or 'v_grad' in some notations for IFT).
    Shape: (S*A,)
    """
    SA = S * A
    grad_Ltrue_Q = np.zeros(SA)

    # First, pre-compute (Q(s,a) - BQ(s,a)) for all (s,a) for efficiency
    # BQ uses the true MDP parameters (r_true, p_true)
    BQ_true = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            reward_term_true = true_r[s, a]
            next_state_values_lse_true = np.array(
                [logsumexp(Q_star_model[s_prime, :]).item() for s_prime in range(S)]
            )
            transition_term_true = np.sum(true_p[s, a, :] * next_state_values_lse_true)
            BQ_true[s, a] = reward_term_true + gamma * transition_term_true

    diff_Q_BQ = Q_star_model - BQ_true  # Element-wise (Q(s,a) - BQ(s,a))

    # Iterate through target Q(s_target, a_target) for which we compute the gradient
    for target_idx in range(SA):
        s_target, a_target = np.unravel_index(target_idx, (S, A))

        # Term 1: 2 * (Q(s_target, a_target) - BQ(s_target, a_target))
        grad_term1 = 2 * diff_Q_BQ[s_target, a_target]

        # Term 2: -2 * sum_sa (Q(s,a) - BQ(s,a)) * ∂BQ(s,a)/∂Q(s_target, a_target)
        grad_term2 = 0.0
        for s in range(S):
            for a in range(A):
                # ∂BQ(s,a)/∂Q(s_target, a_target) is non-zero only if true_p[s,a,s_target] > 0
                # It's gamma * p_true(s_target|s,a) * softmax_Q(s_target, a_target)
                if true_p[s, a, s_target] > 0:
                    softmax_term_true_deriv = np.exp(
                        Q_star_model[s_target, a_target]
                    ) / np.sum(np.exp(Q_star_model[s_target, :]))
                    derivative_part_true = (
                        gamma * true_p[s, a, s_target] * softmax_term_true_deriv
                    )
                    grad_term2 -= 2 * diff_Q_BQ[s, a] * derivative_part_true

        grad_Ltrue_Q[target_idx] = grad_term1 + grad_term2

    return grad_Ltrue_Q


def update_model_parameters_tabular(
    Q_star, r_model, p_model_logits, r_true_mdp, p_true_mdp, gamma_mdp, lr_theta
):
    """
    **Outer Loop Update Function for Tabular Case.**

    Updates the model parameters (r_model, p_model_logits) using the OMD gradient.
    This corresponds to the "Update model parameters θ according to (14)" step in Algorithm 1 [1],
    but for tabular, we use the explicit IFT formulation from (7) [3].

    We minimize Ltrue(Q*) as the objective here, so it's gradient descent.

    Args:
        Q_star (np.array): The Q-function (w*) converged in the inner loop.
        r_model (np.array): Current learned reward function (part of theta).
        p_model_logits (np.array): Current learned transition logits (part of theta).
        r_true_mdp (np.array): True MDP reward function (for Ltrue calculation).
        p_true_mdp (np.array): True MDP transition probabilities (for Ltrue calculation).
        gamma_mdp (float): Discount factor for the true MDP.
        lr_theta (float): Learning rate for the model parameters.

    Returns:
        tuple: (new_r_model, new_p_model_logits, success_flag)
               success_flag indicates if the Jacobian inversion was successful.
    """
    # 1. Compute ∂Ltrue/∂Q* (gradient of the outer objective w.r.t. Q*)
    grad_Ltrue_Q_star = compute_grad_Ltrue_Q(Q_star, r_true_mdp, p_true_mdp, gamma_mdp)

    # 2. Compute ∂f/∂Q (Jacobian of the constraint function w.r.t. Q)
    J_f_Q = compute_jacobian_f_Q(Q_star, r_model, p_model_logits, gamma_mdp)

    # 3. Compute ∂f/∂theta (Jacobian of the constraint function w.r.t. model parameters)
    J_f_theta = compute_jacobian_f_theta(Q_star, r_model, p_model_logits, gamma_mdp)

    # 4. Compute (∂f/∂Q)^-1 (Inverse Jacobian)
    try:
        J_Q_inv = np.linalg.inv(J_f_Q)
    except np.linalg.LinAlgError:
        print("Error: Jacobian ∂f/∂Q is singular. Cannot invert. Model update skipped.")
        return r_model, p_model_logits, False  # Indicate failure

    # 5. Compute ∂Q*/∂theta = - ( (∂f/∂Q)^-1 @ (∂f/∂theta) )
    # This term tells us how Q* (the inner loop solution) changes with respect to theta
    dQ_star_dtheta = -(J_Q_inv @ J_f_theta)  # Shape: (S*A) x (total_theta_params)

    # 6. Compute total gradient ∂Ltrue/∂theta = (∂Ltrue/∂Q*) @ (∂Q*/∂theta)
    # Reshape grad_Ltrue_Q_star to a row vector for matrix multiplication
    total_grad_theta = grad_Ltrue_Q_star.reshape(1, -1) @ dQ_star_dtheta
    total_grad_theta = total_grad_theta.flatten()  # Flatten back to a 1D array

    # Extract gradients for r_model and p_model_logits parts of theta
    num_r_params = S * A
    grad_r_model = total_grad_theta[:num_r_params].reshape(S, A)
    grad_p_logits = total_grad_theta[num_r_params:].reshape(S, A, S)

    # 7. Update model parameters using gradient descent (since we are minimizing Ltrue)
    new_r_model = r_model - lr_theta * grad_r_model
    new_p_logits = p_model_logits - lr_theta * grad_p_logits

    return new_r_model, new_p_logits, True  # Indicate success


# --- Main Training Loop Simulation (Orchestrates Inner and Outer Loops) ---


def run_omd_tabular_training(num_outer_loops, lr_theta):
    """
    Simulates the OMD training process for tabular MDPs.
    This serves as the main execution flow similar to Algorithm 1.
    """
    global r_theta, p_theta_logits  # Access global model parameters

    print("--- Starting OMD Tabular Training ---")
    print(f"Initial r_theta (random):\n{r_theta}")
    print(f"Initial p_theta (from logits):\n{softmax_p_logits(p_theta_logits)}")

    # Initial Q* calculation (warm start for first outer loop iteration)
    current_Q_star = find_Q_star_tabular(r_theta, p_theta_logits, gamma, alpha_softmax)
    initial_L_true = compute_Ltrue(current_Q_star, r_true, p_true, gamma)
    print(f"Initial Q* from model (converged by VI):\n{current_Q_star}")
    print(f"Initial L_true (Bellman error with true MDP): {initial_L_true:.6f}")
    print("-" * 50)

    for outer_loop_iter in range(num_outer_loops):
        print(f"Outer Loop Iteration {outer_loop_iter + 1}/{num_outer_loops}")

        # Step 1: Inner Loop - Find Q* for current model parameters (theta)
        # We warm-start value iteration with the Q* from the previous outer loop
        current_Q_star = find_Q_star_tabular(
            r_theta, p_theta_logits, gamma, alpha_softmax, Q_init=current_Q_star
        )

        # Step 2: Outer Loop - Update model parameters theta using Implicit Differentiation
        updated_r_theta, updated_p_theta_logits, success = (
            update_model_parameters_tabular(
                current_Q_star, r_theta, p_theta_logits, r_true, p_true, gamma, lr_theta
            )
        )

        if not success:  # If Jacobian inversion failed
            print("Training halted due to an issue in model parameter update.")
            break

        # Update global model parameters
        r_theta = updated_r_theta
        p_theta_logits = updated_p_theta_logits

        # Monitor progress
        L_true_current = compute_Ltrue(current_Q_star, r_true, p_true, gamma)
        print(f"  Current L_true: {L_true_current:.6f}")
        # Uncomment below to see intermediate model parameters and Q-values
        # print(f"  Current r_theta:\n{r_theta}")
        # print(f"  Current p_theta (from logits):\n{softmax_p_logits(p_theta_logits)}")
        # print(f"  Current Q*:\n{current_Q_star}")
        print("-" * 50)

    print("\n--- Training Complete ---")
    print(f"Final r_theta:\n{r_theta}")
    print(f"Final p_theta (from logits):\n{softmax_p_logits(p_theta_logits)}")
    final_Q_star = find_Q_star_tabular(
        r_theta, p_theta_logits, gamma, alpha_softmax, Q_init=current_Q_star
    )
    print(f"Final Q*:\n{final_Q_star}")
    print(f"Final L_true: {compute_Ltrue(final_Q_star, r_true, p_true, gamma):.6f}")


# --- Example Usage ---
# Hyperparameters for the training process
NUM_OUTER_LOOPS = 100  # Number of times to update the model parameters (theta)
LEARNING_RATE_THETA = 0.01  # Step size for gradient descent on model parameters

# Execute the training
run_omd_tabular_training(NUM_OUTER_LOOPS, LEARNING_RATE_THETA)

"""
Context on "Standard Methods for Minimizing a Function"
When a source refers to "standard methods for minimizing a function," it typically implies gradient-based optimization algorithms.
• Objective: If you have a function L(x) that you want to minimize, the goal is to find x that makes L(x) as small as possible.
• Gradient: The gradient ∇L(x) is a vector that points in the direction of the steepest increase of the function L at point x. For a multi-variable function L(x_1, x_2, ..., x_n), the gradient is [∂L/∂x_1, ∂L/∂x_2, ..., ∂L/∂x_n].
• Update Rule (Gradient Descent): To minimize L(x), you take steps in the direction opposite to the gradient. The update rule is: x_new = x_old - learning_rate * ∇L(x_old)
    ◦ learning_rate (or alpha in some contexts) is a small positive scalar that determines the size of each step. It's a crucial hyperparameter.
• Gradient Ascent: If your objective is to maximize a function J(x), you would take steps in the direction of the gradient: x_new = x_old + learning_rate * ∇J(x_old)
    ◦ In OMD, if you choose to maximize expected returns J(πQ), you would use gradient ascent. Since this implementation uses Ltrue (Bellman error) as the objective, we perform gradient descent.
In the context of OMD for tabular MDPs, calculating the "gradient" ∇Ltrue(θ) is the main challenge because Ltrue depends on θ implicitly through Q* = ϕ(θ). The Implicit Function Theorem provides the mathematical framework to derive this implicit gradient, allowing us to use standard gradient-based optimization. Your request for "imperative Python code with only numpy" necessitates manually calculating these gradients (Jacobian matrices and vector-Jacobian products) rather than relying on automatic differentiation frameworks (like TensorFlow or PyTorch), which would compute these symbolically or numerically for you.
"""


# %% [code]
import numpy as np
import copy  # Needed for deep copying the q_table

## "Algorithm 1: Model Based RL with OMD Input:"
## "Initial parameters w, θ, empty replay buffer D."

# --- New: Hyperparameters for target Q-table update ---
# Tau (τ) for Exponential Moving Average (EMA) update. A small value (e.g., 0.005 or 0.01) is typical.
# This controls how slowly the target_q_table tracks the q_table.
tau = 0.005

# --- New: Initialization of target_q_table (w̄) ---
# Initialize the primary Q-table (w)
# (Assuming q_table, states, actions, etc. are already defined or imported from elsewhere)
# For demonstration, let's define dummy ones here:
states = ["s1", "s2", "s3"]
actions = ["a1", "a2"]
q_table = {s: {a: np.random.rand() for a in actions} for s in states}
# Initialize the target Q-table (w̄) as a deep copy of the primary Q-table
target_q_table = copy.deepcopy(q_table)

# Dummy models and parameters for the rest of the code to run
max_iterations = 10  # Outer loop iterations
k = 5  # Inner loop optimization steps
gamma = 0.99  # Discount factor
learning_rate = 0.01  # For updating q_table

# Dummy true reward and probability models (for environment interaction and experience collection)
rewards = {
    "s1": {"a1": 1.0, "a2": 0.5},
    "s2": {"a1": -0.1, "a2": 1.0},
    "s3": {"a1": 0.2, "a2": -0.5},
}
probs = {
    "s1": {"a1": {"s1": 0.8, "s2": 0.2}, "a2": {"s1": 0.1, "s2": 0.9}},
    "s2": {"a1": {"s1": 0.5, "s2": 0.5}, "a2": {"s1": 0.9, "s2": 0.1}},
    "s3": {"a1": {"s1": 0.7, "s3": 0.3}, "a2": {"s2": 0.6, "s3": 0.4}},
}

# Dummy model-based reward and probability models (θ)
# These are used to calculate the Bellman target (BθQw̄)
rewards_theta = copy.deepcopy(
    rewards
)  # In a real setting, these would be learned models
probs_theta = copy.deepcopy(probs)  # In a real setting, these would be learned models


def get_softmax_policies(s, actions, q_table):
    """Calculates softmax probabilities for actions given a state and Q-table."""
    q_values_for_state = np.array([q_table[s][a] for a in actions])
    exp_q = np.exp(
        q_values_for_state - np.max(q_values_for_state)
    )  # Numerical stability
    probs = exp_q / np.sum(exp_q)
    return {s: {action: prob for action, prob in zip(actions, probs)}}


def soft_bellman(
    s,
    a,
    probs_model,
    rewards_model,
    all_states,
    all_actions,
    q_values_for_target_network,
    gamma,
):
    """
    Calculates the soft Bellman optimality operator.
    Crucially, it uses the 'q_values_for_target_network' (representing Qw̄)
    for the Q-values of the next states.
    """
    reward_term = rewards_model[s][a]

    expected_log_sum_exp = 0.0
    # Iterate over possible next states and their probabilities from the *model*
    for s_prime_next, prob_s_prime_next in probs_model[s][a].items():
        # Get Q-values for all actions in the next state using the *target_q_table*
        q_values_next_state = np.array(
            [
                q_values_for_target_network[s_prime_next][a_prime]
                for a_prime in all_actions
            ]
        )

        # Calculate log-sum-exp using the trick for numerical stability
        max_q = np.max(q_values_next_state)
        log_sum_exp_term = max_q + np.log(np.sum(np.exp(q_values_next_state - max_q)))

        expected_log_sum_exp += prob_s_prime_next * log_sum_exp_term

    return reward_term + gamma * expected_log_sum_exp


def update_q_table(q_table_to_update, sampled_s, sampled_a, bellman_target, lr):
    """
    Updates Qw parameters w to minimize L(θ, w).
    This function modifies the 'q_table_to_update' (your primary Q-table) in place.
    """
    current_q_value = q_table_to_update[sampled_s][sampled_a]
    # Simple gradient descent step for the squared Bellman error
    error = current_q_value - bellman_target
    q_table_to_update[sampled_s][sampled_a] -= lr * error


# --- Main RL loop ---
d = {}  # Empty replay buffer
for ir in range(0, max_iterations):
    ## "Set s to be the current state."
    if ir == 0:
        s = np.random.choice(states)
    else:
        s = s_prime  # s_prime from the previous environment interaction

    ## "Sample an action a using softmax over Qw(s, a)."
    # Note: This uses the current (online) q_table (w) for policy generation
    action_probs = get_softmax_policies(s, actions, q_table)
    current_state_policies = action_probs[s]
    a = np.random.choice(actions, p=list(current_state_policies.values()))

    ## "Apply a to get r = r(s, a), s′ ∼ p(s′|s, a)."
    # These come from the true environment dynamics/rewards
    r = rewards[s][a]
    current_trans_probs = probs[s][a]
    potential_next_states = list(current_trans_probs.keys())
    s_prime = np.random.choice(
        potential_next_states, p=list(current_trans_probs.values())
    )

    # "Append (s, a, s′, r) to buffer D."
    # The replay buffer stores experience from interaction with the *true* environment.
    d[ir] = {"s": s, "a": a, "r": r, "s_prime": s_prime}

    ## for i = 1 to K do
    ## "We make K steps of an optimization method to approximate w∗ = φ(θ) where K is
    ## a hyperparameter and reuse the weights from the previous outer loop iterations."
    for ik in range(0, k):  # Loop for K optimization steps of Qw
        ## "Sample (s, a) from buffer D."
        # This experience (s, a) is from the true environment.
        d_index = np.random.choice(list(d.keys()))
        d_entry = d[d_index]
        ds = d_entry["s"]
        da = d_entry["a"]

        ## "Apply θ to get r = rθ(s, a), s′ ∼ pθ(s′|s, a)."
        # This step implies that the Bellman target uses the *model's* (θ)
        # predictions for rewards and next state probabilities.

        ## "Update Qw parameters w to minimize L(θ, w)."
        ## "BθQ(s, a) = rθ(s, a) + γEpθ(s′|s,a) log ∑ a′ expQ(s′, a′)"

        # *** IMPORTANT CHANGE HERE: Pass target_q_table (w̄) to soft_bellman ***
        # The Bellman target calculation uses the *stable* Q-values from the target network (w̄)
        q_bellman = soft_bellman(
            ds, da, probs_theta, rewards_theta, states, actions, target_q_table, gamma
        )

        # Update the primary Q-table (w) using the calculated Bellman target
        update_q_table(q_table, ds, da, q_bellman, learning_rate)

    # --- New: Update target_q_table (w̄) after K inner loop steps ---
    # This step is performed once per outer loop iteration (after the K updates to q_table).
    # It updates the target_q_table (w̄) to slowly track the primary q_table (w)
    # using an Exponential Moving Average (EMA).
    # This ensures that during the K inner loop steps of the *next* outer iteration,
    # the target_q_table remains relatively fixed and stable.
    for s_key in states:
        for a_key in actions:
            # target_q_table_new_value = (1 - tau) * target_q_table_old_value + tau * q_table_current_value
            target_q_table[s_key][a_key] = (1 - tau) * target_q_table[s_key][
                a_key
            ] + tau * q_table[s_key][a_key]

    ## "Update model parameters θ according to (14)."
    # This part of Algorithm 1 (updating theta) is not implemented in your snippet.
    # It would typically involve using the gradients derived from the implicit differentiation (IFT).

print("Simulation complete.")
print("\nFinal Q-Table (w):")
for s_key, actions_dict in q_table.items():
    print(f"  {s_key}: {actions_dict}")

print("\nFinal Target Q-Table (w̄):")
for s_key, actions_dict in target_q_table.items():
    print(f"  {s_key}: {actions_dict}")
