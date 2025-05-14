AFRAME.registerComponent('planes', {
  init() {

    for (let ii = 0; ii < 10000; ii++) {
      this.createSphere(ii)
    }
  },

  createSphere(index) {
    const sphere = document.createElement('a-entity')
    const colorString = `#${Math.floor(Math.random()*4096).toString(16).padStart(3, "0")}`
    sphere.setAttribute("instanced-mesh-member", { mesh: "#mesh", 
                                                   colors: colorString })
    const p = sphere.object3D.position
    p.set(-5 + 0.02 * index, 0, 0)

    this.el.appendChild(sphere)
  }
})