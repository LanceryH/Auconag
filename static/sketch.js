let create_clicked = false;
let run_clicked = false;
let freq_sim = 0;
let freq_aff = 0;

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
        sat_pos_ini = data;
        create_clicked = true;
    });
    socket.on('801', function(data) {
        sat_pos_fin = data;
        run_clicked = true;
    });
    socket.on('802', function(data) {
        freq_sim = data;
    });
    socket.on('803', function(data) {
        freq_aff = data;
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
    scale(0.03);

    push();
    texture(img);
    strokeWeight(0);
    rotateX(0);
    rotateY(-180);
    rotateZ(0);
    sphere(6371, 24, 24);
    pop();

    if (create_clicked) {
        push();
        strokeWeight(16);
        stroke(255,0,0);
        beginShape(POINTS);
        vertex(sat_pos_ini[0], sat_pos_ini[1], sat_pos_ini[2]);
        endShape();
        pop();
    }

    if (run_clicked) {
        push();
        strokeWeight(16);
        stroke(0,255,0);
        beginShape(POINTS);
        vertex(sat_pos_fin[0], sat_pos_fin[1], sat_pos_fin[2]);
        endShape();
        pop();
    }

    document.getElementById("sim").textContent=freq_sim;
    document.getElementById("aff").textContent=freq_aff;
}

function windowResized() {
    let width_live = document.getElementById('contain').clientWidth-290;
    let height_live = document.getElementById('contain').clientHeight;
    resizeCanvas(width_live, height_live);
  }