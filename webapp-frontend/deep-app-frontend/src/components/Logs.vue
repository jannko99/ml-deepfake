<template>
  <div class="container-fluid">
    <br>

    <div v-if="tensorboardUrl" class="ratio ratio-16x9">
      <iframe :src="tensorboardUrl" allowfullscreen></iframe>
    </div>
    <div v-else>
      Loading TensorBoard...
    </div>


  </div>
</template>

<script>
export default {
  data() {
    return {
      tensorboardUrl: ''
    };
  },
  created() {
    fetch('/tensorboard')
      .then(response => response.json())
      .then(data => {
        console.log('TensorBoard started');
        this.tensorboardUrl = data.url;
      })
      .catch(error => {
        console.error('Error starting TensorBoard:', error);
      });
  }
};
</script>
