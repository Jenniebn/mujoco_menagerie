import mujoco as mj
import glfw
import numpy as np
import OpenGL.GL as gl

def main():
    # Load the model
    m = mj.MjModel.from_xml_path('mjx_single_cube.xml')
    d = mj.MjData(m)

    # Initialize GLFW and create window
    if not glfw.init():
        print("Could not initialize GLFW")
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1200, 900, "Panda Robot Pickup Simulation", None, None)
    if not window:
        glfw.terminate()
        print("Could not create GLFW window")
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Enable v-sync
    glfw.swap_interval(1)

    # Initialize visualization data structures
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)

    # For perturbations (optional, but often used in interactive simulations)
    pert = mj.MjvPerturb()
    mj.mjv_defaultPerturb(pert)

    # Visualization options
    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)

    # Enable additional visualization flags
    opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    # Rendering context
    con = mj.MjrContext(m, mj.mjtFontScale.mjFONTSCALE_100)

    # Create scene
    scn = mj.MjvScene(m, maxgeom=1000)

    # Set camera parameters
    cam.distance = 3.0
    cam.azimuth = 90.0
    cam.elevation = -20.0

    # Set initial keyframe (home position)
    home_key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, 'home')
    mj.mj_resetDataKeyframe(m, d, home_key)

    # Set pickup keyframe
    # pickup_key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, 'pickup1')
    # mj.mj_resetDataKeyframe(m, d, pickup_key)

    # Main simulation loop
    while not glfw.window_should_close(window):
        # Simulation start time
        simstart = d.time

        # Simulate for 1/60 of a second
        while d.time - simstart < 1.0/60.0:
            mj.mj_step(m, d)

        # Get framebuffer viewport
        width, height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, width, height)

        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Update the scene
        mj.mjv_updateScene(
            m, 
            d, 
            opt, 
            None, 
            cam, 
            mj.mjtCatBit.mjCAT_ALL, 
            scn
        )

        # Render the scene
        mj.mjr_render(viewport, scn, con)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

        # Optional: print object positions (you can remove or comment out)
        box_body_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'box')
        end_effector_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'panda_hand')
        
        print(f"Box Position: {d.xpos[box_body_id]}")
        print(f"End Effector Position: {d.xpos[end_effector_id]}")

    # Cleanup
    glfw.terminate()
    mj.mjv_freeScene(scn)
    mj.mjr_freeContext(con)

if __name__ == "__main__":
    main()
