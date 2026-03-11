within DemoPlant;
model SingleZoneThermal
  "Single-zone synthetic thermal model corresponding to paper verification dynamics"
  parameter Real alpha = 0.08 "Ambient coupling";
  parameter Real beta = 0.35 "Cooling actuation gain";
  parameter Real gamma = 0.6 "Disturbance gain";
  parameter Real x0 = -1.5 "Initial state";

  input Real u "Cooling command in [0,10]";
  input Real Tamb "Ambient temperature input";
  input Real xi "Uncertain load/disturbance";
  output Real x(start=x0) "Zone thermal safety state";

equation
  // Discrete-time paper model x_{t+1}=x_t+alpha(Tamb-x_t)-beta*u+gamma*xi
  // is represented in continuous simulation as Euler-equivalent derivative form.
  der(x) = alpha * (Tamb - x) - beta * u + gamma * xi;
end SingleZoneThermal;
