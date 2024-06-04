<template>
    <nav>
      <router-link to="/">Model</router-link>
      <router-link to="/logs">TensorBoard Logs</router-link>
    </nav>
  </template>
  
  <script>
  export default {
    name: 'Navbar'
  }
  </script>
  
  <style>
  nav {
    display: flex;
    justify-content: space-around;
    background-color: #333;
    padding: 1em;
  }
  
  a {
    color: white;
    text-decoration: none;
  }
  
  a:hover {
    text-decoration: underline;
  }
  </style>
  