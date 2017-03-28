#!/usr/bin/env Rscript
# Evaluate the results of CTR using recall.
dyn.load("utils.so")

library(plyr)

fold <- 5
breaks <- seq(20, 200, 20)

read.user <- function(filename, offset=1) {
  one <- scan(filename, what = "", sep = "\n", quiet=T)
  two <- strsplit(one, " ", fixed = TRUE)
  lapply(two, function(x) (offset + as.vector(as.integer(x[-1]))))
}


summary.recall <- function(root.path) {
  users <- read.user(sprintf("%s/users.dat", root.path))
  nbibs <- data.frame(user.id=1:length(users), total=sapply(users, length))
  
  paths <- sort(list.files(root.path, pattern="cv-cf-[1-5]"))
  recall <- NULL
  for (i in seq(fold)) {
    path <- paths[i]
    recall.tmp <- read.csv(sprintf("%s/%s/recall-user.dat", root.path, path))
    recall <- rbind(recall, data.frame(method="cf-in-matrix", recall.tmp))
  }

  #ctr + cf
  paths <- sort(list.files(root.path, pattern="cv-ctr-[1-5]-cf"))
  for (i in seq(fold)) {
    path <- paths[i]
    recall.tmp <- read.csv(sprintf("%s/%s/recall-user.dat", root.path, path))
    recall <- rbind(recall, data.frame(method="ctr-in-matrix", recall.tmp))
  }

  #ctr + ofm
  paths <- sort(list.files(root.path, pattern="cv-ctr-[1-5]-ofm"))
  for (i in seq(fold)) {
    path <- paths[i]
    recall.tmp <- read.csv(sprintf("%s/%s/recall-user.dat", root.path, path))
    recall <- rbind(recall, data.frame(method="ctr-out-matrix", recall.tmp))
  }
  recall <- merge(recall, nbibs)
  recall$recall <- fold*recall$recall/recall$total
  recall <- ddply(recall, .(method, N), function(df) mean(df$recall))
  names(recall) <- c("method", "N", "recall")
  plot(recall[1:10,][,2],recall[1:10,][,3], pch=21, col="red", bg="red")
  lines(recall[1:10,][,2],recall[1:10,][,3], pch=21, col="red", bg="red")
  points(recall[11:20,][,2],recall[11:20,][,3], pch=24, col="green", bg="green")
  lines(recall[11:20,][,2],recall[11:20,][,3], pch=24, col="green", bg="green")
  recall
}

compute.recall.all <- function(root.path) {
  cat(sprintf("reading from %s\n", root.path))
  splits.cf.path <- sprintf("%s/splits.cf.dat", root.path)
  cat(sprintf("loading %s ...\n", splits.cf.path)) 
  load(splits.cf.path)

  #cf
  paths <- sort(list.files(root.path, pattern="cv-cf-[1-5]"))
  cf.compute.recall(root.path, paths, splits)

  #ctr + cf
  paths <- sort(list.files(root.path, pattern="cv-ctr-[1-5]-cf"))
  cf.compute.recall(root.path, paths, splits)

  #ctr + ofm
  paths <- sort(list.files(root.path, pattern="cv-ctr-[1-5]-ofm"))
  ofm.compute.recall(root.path, paths)
}

cf.compute.recall <- function(root.path, paths, splits, top=200) {
  for (i in seq(fold)) {
    path <- paths[i] 
    score.path <- sprintf("%s/%s/score.dat", root.path, path)
    load(score.path)
    num.users <- nrow(score)

    user.test <- read.user(sprintf("%s/cf-test-%d-users.dat", root.path, i))

    recall.user.file <- sprintf("%s/%s/recall-user.dat", root.path, path)
    cat(sprintf("Computing %s...\n", recall.user.file))
    write("user.id,fold,N,recall", file=recall.user.file)

    for (j in 1:num.users) {
       user.items <- user.test[[j]]
       ids.left <- splits[[j]][[i]]
       score.u <- score[j, ids.left] 
       s <- sort.int(x=score.u, decreasing=TRUE, index.return=TRUE)
       idx <- s$ix[1:top]
       item.ids <- ids.left[idx]
       intest <- as.integer(sapply(item.ids, FUN=function(x) {x %in% user.items}))
       recall <- cumsum(intest)
       recall <- recall[breaks] # unnormalized
       write(rbind(j, i, breaks, recall), ncolumns=4, file=recall.user.file, append=T, sep=",")
    }
  }
}

ofm.compute.recall <- function(root.path, paths, top=200) {
  for (i in seq(fold)) {
    path <- paths[i] 
    score.path <- sprintf("%s/%s/score.dat", root.path, path)
    load(score.path)
    num.users <- nrow(score)

    user.test <- read.user(sprintf("%s/ofm-test-%d-users.dat", root.path, i))
    heldout.set <- c()
    for (u in user.test) heldout.set <- union(heldout.set, u)
    heldout.set <- sort(heldout.set)
    score <- score[,heldout.set]

    recall.user.file <- sprintf("%s/%s/recall-user.dat", root.path, path)
    cat(sprintf("Computing %s...\n", recall.user.file))
    write("user.id,fold,N,recall", file=recall.user.file)

    for (j in 1:num.users) {
       user.items <- user.test[[j]]
       score.u <- score[j,] 
       s <- sort.int(x=score.u, decreasing=TRUE, index.return=TRUE)
       idx <- s$ix[1:top]
       item.ids <- heldout.set[idx]
       intest <- as.integer(sapply(item.ids, FUN=function(x) {x %in% user.items}))
       recall <- cumsum(intest)
       recall <- recall[breaks] # unnormalized
       write(rbind(j, i, breaks, recall), ncolumns=4, file=recall.user.file, append=T, sep=",")
    }
  }
}

scoring <- function(path, num.topics) {
  u.path <- sprintf("%s/final-U.dat", path)
  u <- scan(u.path)
  u <- matrix(u, nrow=num.topics)

  v.path <- sprintf("%s/final-V.dat", path)
  v <- scan(v.path)
  v <- matrix(v, nrow=num.topics)
  score <- t(u) %*% v
  score.path <- sprintf("%s/score.dat", path)
  cat(sprintf("computing score and saving to %s ...\n", score.path))
  save(score, file=score.path, compress=T)
}

scoring.all <- function(root.path, num.topics) {
  paths <- list.files(root.path, pattern="cv-")
  for (path in paths) {
    result.path <- paste(root.path, path, sep="/")
    cat(sprintf("scoring %s ...\n", result.path))
    scoring(result.path, num.topics)
  }
}
