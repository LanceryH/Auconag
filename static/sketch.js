function preload() {
    img = loadImage(imagePath); // Use the image path passed from HTML
}

function setup() {
    let width = document.getElementById('sketch').clientWidth;
    let height = document.getElementById('sketch').clientHeight;

    const canvas = createCanvas(width, height, WEBGL);
    canvas.parent('sketch');
    background(220);

    socket = io(); 
    socket.on('send_data', function(data) {
        names = data["Names"];
        colors = data["Colors"];
        sat_data = data;
    });

    const button = document.getElementById('run');
    button.addEventListener('click', () => {
      alert('Button clicked!');
      socket.emit('button_clicked', 'Button clicked!');
    });
}

function draw() {
    background(0);
    orbitControl(1, 1, 1);
    scale(0.02);

    push();
    texture(img);
    strokeWeight(0);
    rotateX(0);
    rotateY(-180);
    rotateZ(0);
    sphere(6371, 24, 24);
    pop();
}

function windowResized() {
    let width_live = document.getElementById('sketch').clientWidth;
    let height_live = document.getElementById('sketch').clientHeight;
    resizeCanvas(width_live, height_live);
  }