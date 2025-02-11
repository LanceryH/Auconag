let sat_state = false;
let sat_state_prop = false;

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
    socket.on('800', function(data) {
        sat_pos = data;
        sat_state = true;
    });

    socket = io(); 
    socket.on('801', function(data) {
        sat_pos_prop = data;
        sat_state_prop = true;
    });

    const button_create = document.getElementById('create');
    button_create.addEventListener('click', () => {
      //alert('Create');
      socket.emit('button_create', 'Button clicked!');
    });

    const button_run = document.getElementById('run');
    button_run.addEventListener('click', () => {
      //alert('Button clicked!');
      socket.emit('button_run', 'Button clicked!');
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

    if (sat_state) {
        push();
        strokeWeight(16);
        stroke(255,0,0);
        beginShape(POINTS);
        vertex(sat_pos[0], sat_pos[1], sat_pos[2]);
        endShape();
        pop();
    }

    if (sat_state_prop) {
        push();
        strokeWeight(16);
        stroke(0,255,0);
        beginShape(POINTS);
        vertex(sat_pos_prop[0], sat_pos_prop[1], sat_pos_prop[2]);
        endShape();
        pop();
    }
}

function windowResized() {
    let width_live = document.getElementById('sketch').clientWidth;
    let height_live = document.getElementById('sketch').clientHeight;
    resizeCanvas(width_live, height_live);
  }