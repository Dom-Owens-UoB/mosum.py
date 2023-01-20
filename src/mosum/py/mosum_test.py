import numpy as np
#' Help function: asymptotic scaling.
def asymptoticA(x: float): return np.sqrt(2*np.log(x))

#' Help function: asymptotic shift
def asymptoticB(x: float, K:int): return 2*np.log(x) + 0.5*np.log(np.log(x)) + np.log((K**2+K+1)/(K+1)) - 0.5*np.log(np.pi)


#' MOSUM asymptotic critical value
#'
#' Computes the asymptotic critical value for the MOSUM test.
#' @param n an integer value for the length of the input data
#' @param G.left,G.right integer values for the left and right moving sum bandwidth (G.left, G.right)
#' @param alpha a numeric value for the significance level with
#' \code{0 <= alpha <= 1}
#' @return a numeric value for the asymptotic critical value for the MOSUM test
#' @examples
#' x <- testData(lengths = rep(100, 3), means = c(0, 5, -2), sds = rep(1, 3), seed = 1234)$x
#' m <- mosum(x, G = 40)
#' par(mfrow = c(2, 1))
#' plot(m$stat, type = "l", xlab = "Time", ylab = "", main = "mosum")
#' abline(h = mosum.criticalValue(300, 40, 40, .1), col = 4)
#' abline(v = m$cpts, col = 2)
#' plot(m, display = "mosum") # identical plot is produced
#' @export
def criticalValue(n, G_left, G_right, alpha):
  G_min = min(G_left, G_right)
  G_max = max(G_left, G_right)
  K = G_min / G_max
  return (asymptoticB(n/G_min,K) - np.log(np.log(1/np.sqrt(1-alpha))))/asymptoticA(n/G_min)


#' MOSUM asymptotic p-value
#'
#' Computes the asymptotic p-value for the MOSUM test.
#' @param z a numeric value for the observation
#' @param n an integer value for the length of the input data
#' @param G.left,G.right integer values for the left moving sum bandwidth (G.left,G.right)
#' @return a numeric value for the asymptotic p-value for the asymmetric MOSUM test
#' @keywords internal
def pValue(z, n, G_left, G_right=None):
  if G_right is None: G_right = G_left
  G_min = min(G_left, G_right)
  G_max = max(G_left, G_right)
  K = G_min / G_max
  return 1-np.exp(-2*np.exp(asymptoticB(n/G_min,K) - asymptoticA(n/G_min)*z))

