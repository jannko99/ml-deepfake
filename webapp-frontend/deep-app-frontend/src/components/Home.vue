<template>
  <div class="container-fluid text-center">
    <br>
    <form @submit.prevent="submitFile">
      <input class="form-control border-success-subtle form-control-lg" type="file" @change="onFileChange" />
      <div class="row align-items-start justify-content-evenly">
        <div class="col">
          <div v-if="imageSrc">
            <img :src="imageSrc" class="rounded mx-auto d-block" width="300" height="300" alt="Uploaded Image" />
          </div>
        </div>
        <div class="col">
          <br>
          <div class="d-grid gap-2">
            <button type="submit" class="btn btn-success btn-lg">Predict</button>
          </div>
          <br>
          <table class="table table-striped-columns table-dark table-hover table-bordered">
            <thead>
              <tr>
                <th colspan="2">Prediction</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Image is:</th>
                <td>{{ prediction_text }}</td>
              </tr>
              <tr>
                <th scope="row">Number:</th>
                <td>{{ prediction_percent }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </form>
  </div>
</template>

<script>
export default {
  data() {
    return {
      file: null,
      imageSrc: null,
      prediction_percent: null,
      prediction_text: null
    };
  },
  methods: {
    onFileChange(e) {
      const file = e.target.files[0];
      this.file = file;
      this.imageSrc = URL.createObjectURL(file);
    },
    async submitFile() {
      const formData = new FormData();
      formData.append('file', this.file);

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      this.prediction_percent = data.prediction;

      if ( data.prediction > 0.5){
          this.prediction_text = "Real"
      } else {
        this.prediction_text = "Fake"
      }
    }
  }
};
</script>

<style>
img {
  max-width: 300px;
  margin-top: 20px;
}
</style>