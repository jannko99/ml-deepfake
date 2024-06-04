import { createRouter, createWebHistory } from 'vue-router'
import Home from './components/Home.vue'
import Logs from './components/Logs.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/logs',
    name: 'Logs',
    component: Logs
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
