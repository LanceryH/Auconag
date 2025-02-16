let create_clicked = false;
let run_clicked = false;
let freq_sim = 0;
let freq_aff = 0;

function preload() {
    img_earth = loadImage(imagePath_earth); 
    img_moon = loadImage(imagePath_moon); 

}

function setup() {
    let width = document.getElementById('sketch').clientWidth;
    let height = document.getElementById('sketch').clientHeight;

    const canvas = createCanvas(width, height, WEBGL);
    canvas.parent('sketch');
    background(220);

    socket = io(); 
    socket.on('700', function(data) {
        agent_pos = data;
    });
    socket.on('701', function(data) {
        earth_pos = data;
    });
    socket.on('702', function(data) {
        moon_pos = data;
        run_clicked = true;
    });
    socket.on('800', function(data) {
        freq_sim = data;
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
    const button_test = document.getElementById('test');
    button_test.addEventListener('click', () => {
      //alert('Button clicked!');
      socket.emit('button_test', 'Button clicked!');
    });
}

function draw() {
    background(0);
    orbitControl(1, 1, 1);
    scale(0.02);
    if (run_clicked) {
        push();
        texture(img_earth);
        strokeWeight(0);
        rotateX(0);
        rotateY(-180);
        rotateZ(0);
        translate(earth_pos[0]/1e3, earth_pos[1]/1e3, earth_pos[2]/1e3);
        sphere(6371, 24, 24);
        pop();
        
        push();
        texture(img_moon);
        strokeWeight(0);
        rotateX(0);
        rotateY(-180);
        rotateZ(0);
        translate(moon_pos[0]/1e3, moon_pos[1]/1e3, moon_pos[2]/1e3);
        sphere(6371, 24, 24);
        pop();
        
        push();
        strokeWeight(16);
        stroke(0,255,0);
        beginShape(POINTS);
        vertex(agent_pos[0]/1e3, agent_pos[1]/1e3, agent_pos[2]/1e3);
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