---
author: David Abimbola
pubDatetime: 2023-08-14T16:35:00Z
title: A Quick Introduction to Computational Graphs
postSlug: a-quick-introduction-to-computational-graphs
featured: true
draft: false
tags:
  - graphs
ogImage: ""
description: Understanding graphs and their applications in modern software engineering
---

# A Quick Introduction to Computational Graphs.

## **Types of Graph Traversals**

There are two basic techniques used for graph traversal:

1. Breadth First Search (BFS)
2. Depth First Search (DFS)

In order to understand these algorithms, we will have to view graphs from a slightly different perspective.

Any traversal needs a starting point, but a graph does not have a linear structure like lists or stacks. So how do we give graph traversal a better sense of direction?

This is where the concept of **levels** is introduced. Take any vertex as the starting point. This is the lowest level in your search. The **next level** consists of all the vertices adjacent to your vertex. A level higher would mean the vertices adjacent to these nodes.

With this is in mind, let’s begin our discussion on the two graph traversal algorithms.

### **1. Breadth First Search**

The BFS algorithm earns its name because it grows breadth-wise. All the nodes in a certain level are traversed before moving on to the next level.

The level-wise expansion ensures that for any starting vertex, you can reach all others, one level at a time.

Let’s look at the BFS algorithm in action:
