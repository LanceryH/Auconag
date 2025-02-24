let simulation_state_received = false;
let simulation_infos_received = false;
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
        simulation_state = data;
        simulation_state_received = true;
    });
    socket.on('800', function(data) {
      simulation_infos = data;
      simulation_infos_received = true;
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
    lights();
    orbitControl(1, 1, 1);
    scale(2);

    if (simulation_state_received) {
      for (agent of simulation_state["agents"]){
        push();
        strokeWeight(16);
        stroke(0,255,0);
        beginShape(POINTS);
        vertex(agent[0]/1e5, 
               agent[1]/1e5, 
               agent[2]/1e5);
        endShape();
        pop();
      }

        push();
        texture(img_earth);
        strokeWeight(0);
        rotateX(0);
        rotateY(-180);
        rotateZ(0);
        translate(simulation_state["earth"][0]/1e5, 
                  simulation_state["earth"][1]/1e5, 
                  simulation_state["earth"][2]/1e5);
        sphere(63.71, 24, 24);
        pop();
        
        push();
        texture(img_moon);
        strokeWeight(0);
        rotateX(0);
        rotateY(-180);
        rotateZ(0);
        translate(simulation_state["moon"][0]/1e5, 
                  simulation_state["moon"][1]/1e5, 
                  simulation_state["moon"][2]/1e5);
        sphere(63.71, 24, 24);
        pop();
    }

    if (simulation_infos_received) {
      document.getElementById("sim").textContent=simulation_infos["freq_sim_max"];
      document.getElementById("aff").textContent=simulation_infos["live_sim"];
      document.getElementById("nbage").textContent=simulation_infos["nb_agents"];
    }
}

function windowResized() {
    let width_live = document.getElementById('contain').clientWidth-290;
    let height_live = document.getElementById('contain').clientHeight;
    resizeCanvas(width_live, height_live);
  }
  