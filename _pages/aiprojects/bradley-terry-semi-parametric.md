---
title: "Semi-parametric Bradley–Terry Models with Structural Deep Learning"
collection: aiprojects
permalink: /aiprojects/bradley-terry-semi-parametric/
excerpt: "Walkthrough of an R implementation for semi-parametric Bradley–Terry models using structural debiased machine learning."
author_profile: true
---

## Overview

This tutorial walks through an R implementation of semi-parametric Bradley–Terry models using a structural debiased machine learning approach.

The pipeline:

1. Builds Bradley–Terry preference and cost designs from pairwise comparison data  
2. Fits neural networks for heterogeneous preferences and costs using `torch` in R  
3. Learns conditional Hessians and constructs influence-function style corrections  
4. Delivers point estimates, standard errors, and uniform confidence bands for:
   - Preference parameters  
   - Probabilities of being the best model  
   - Cost parameters  
   - Cost times probability best  
5. Illustrates heterogeneity by token length, prompt similarity, and simple policy trees  

You can download the full R script here:

```text
/bradley_terry_semiparametric.R
