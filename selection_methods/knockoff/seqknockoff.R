# Sampler functions from https://github.com/kormama1/seqknockoff

knockoffs_seq <- function(X, seq_simulator = sim_EN, alpha=0.5, nfolds=5,
                         nlambda=50, parallel=FALSE, lambda.min.ratio=NULL) {
  knockoffs <- X
  names(knockoffs) <- paste0(names(knockoffs),".tilde")
  shf <- sample(ncol(X))
  loop.count <- 1
  for (i in shf) {
    y <- X[[i]] # i-th column serves as response
    Xp <- X[,-i] # columns[-i] serve as predictors
    if (loop.count > 1) Xp <- cbind(knockoffs[,shf[1:(loop.count-1)]], Xp)
    knockoffs[[i]] <- seq_simulator(
      y=y, X=Xp, alpha=alpha, nfolds=nfolds, nlambda=nlambda,
      parallel=parallel, lambda.min.ratio=lambda.min.ratio
    )
    loop.count <- loop.count + 1
  }
  names(knockoffs) <- gsub(".tilde","", names(knockoffs))
  return(knockoffs)
}
sim_EN <- function(y, X, alpha, nfolds, nlambda,
                   parallel, lambda.min.ratio) {
  x <- model.matrix(~., data = X)[,-1]
  options <- list(y=y, x=x, family=if (is.factor(y)) "multinomial" else "gaussian",
                  intercept=TRUE, alpha=alpha, nfolds=nfolds,
                  nlambda=nlambda, parallel=parallel)
  if (!is.null(lambda.min.ratio)) options$lambda.min.ratio <- lambda.min.ratio
  gm.cv <- do.call(glmnet::cv.glmnet, options)
  if (is.factor(y)) {
    classes <- levels(y)
    K <- length(classes)
    beta.coefs <- as.numeric(coef(gm.cv, s = "lambda.min")[[2]])[-1]
    mu <- predict(gm.cv, newx=x, type="response", s="lambda.min")
    mat.multinom <- apply(mu, 1, function(prob) rmultinom(n=1, size=1, prob=prob))
    y.sim <- classes[apply((1:K)*mat.multinom, 2, max)]
    y.sim <- factor(y.sim, levels=classes)
    rmse <- NULL
  } else {
    beta.coefs <- as.numeric(coef(gm.cv, s = "lambda.min"))[-1]
    non.zero.cols <- which(beta.coefs != 0)
    s.lambda = length(non.zero.cols) + 1
    mu <- predict(gm.cv, newx=x, type="response", s="lambda.min")
    rmse = sqrt(sum((y-mu)^2)/(length(y) - s.lambda))
    y.sim <- rnorm(n=length(y), mean=mu, sd=rmse)
  }
  return(y.sim)
}

main <- function(args) {
  X <- read.csv(args[[1]], colClasses="numeric", check.names=FALSE)
  n_levels <- as.integer(strsplit(args[[3]], ",", fixed=TRUE)[[1]])
  categorical <- which(n_levels > 0L)
  for (j in categorical) X[[j]] <- factor(X[[j]], levels=seq_len(n_levels[[j]]))

  n_jobs <- as.integer(args[[8]])
  if (n_jobs > 1L) {
    cluster <- parallel::makeCluster(n_jobs)
    on.exit({parallel::stopCluster(cluster); foreach::registerDoSEQ()}, add=TRUE)
    doParallel::registerDoParallel(cluster)
  }
  set.seed(as.integer(args[[4]]))
  Xk <- knockoffs_seq(
    X, alpha=as.numeric(args[[5]]), nfolds=as.integer(args[[6]]),
    nlambda=as.integer(args[[7]]), parallel=n_jobs > 1L,
    lambda.min.ratio=if (nzchar(args[[9]])) as.numeric(args[[9]]) else NULL
  )
  for (j in categorical) Xk[[j]] <- as.integer(Xk[[j]])
  write.csv(Xk, args[[2]], row.names=FALSE)
}

if (sys.nframe() == 0L) main(commandArgs(trailingOnly=TRUE))
