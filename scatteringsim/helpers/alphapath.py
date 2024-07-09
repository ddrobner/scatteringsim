from scatteringsim.helpers.stoppingpower import stp_interp

def gen_alpha_path(e_0, epsilon=0.1, stepsize=0.001) -> list[float]:
    e_i = e_0
    alpha_path = []
    while e_i > epsilon:
        alpha_path.append(e_i)
        e_i = e_i - stp_interp(e_i)*stepsize
    return alpha_path