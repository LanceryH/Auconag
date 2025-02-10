let socket;
let names = []; 
let sat_data;
let orbit_status = 0;
let selected_status;

function preload() {
    // img = loadImage(imagePath); // Use the image path passed from HTML
}

function setup() {
    cnv = createCanvas(windowWidth, windowHeight, WEBGL); // Create a drawing canvas
    cnv.position(0, 0, "fixed");
    angleMode(DEGREES);
    socket = io(); 

    socket.on('send_data', function(data) {
        names = data["Names"];
        colors = data["Colors"];
        sat_data = data;
    });
}

function draw() {
    console.log(orbit_status)
    resizeCanvas(windowWidth, windowHeight)

    clear();
    background(0);
    orbitControl(1, 1, 1);
    scale(0.02);
    perspective(180 / 3.0, width / height, 0.1, 15000000);

    push();
    // texture(img);
    strokeWeight(1);
    noFill();
    rotateX(0);
    rotateY(-180);
    rotateZ(0);
    stroke(255,0,0)
    sphere(6371, 24, 24);
    pop();  
   
}
