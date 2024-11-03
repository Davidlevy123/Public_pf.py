from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase

from . util import rotateQuaternion, getHeading
import numpy as np
import math
weights_global = []

class PFLocaliser(PFLocaliserBase):
       
    def __init__(self, logger, clock):
        # ----- Call the superclass constructor
        super().__init__(logger, clock)
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.02  # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.01  # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.02  # Odometry model y axis (side-to-side) noise
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predictc
        self.particle_count = 100  # Number of particles
        self.confidence_threshold_upper = 0
        self.confidence_threshold_verylow = 1
       

        self.variance_window = []
        self.variance_window_size = 5
        self.same_confidence_status_count = 0
    def initialise_particle_cloud(self, initialpose):

        
        self.particlecloud = PoseArray()
        self.particlecloud.header.frame_id = "map"
        
        for i in range(self.particle_count):
            # Add Gaussian noise to initial pose
            x_noise = np.random.normal(0, self.ODOM_TRANSLATION_NOISE)
            y_noise = np.random.normal(0, self.ODOM_DRIFT_NOISE)
            theta_noise = np.random.normal(0, self.ODOM_ROTATION_NOISE)


            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + x_noise
            pose.position.y = initialpose.pose.pose.position.y + y_noise
            pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, theta_noise)

            self.particlecloud.poses.append(pose)

        
        #with open("weight_variance.txt", "a") as file:
            
            #file.write("START OF FILE\n")
        
        
        return self.particlecloud

 
    
    def update_particle_cloud(self, scan):
        weights = []

        # Calculate weights for each particle
        for particle in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, particle)
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            return  # Skip if no valid weights

        global weights_global
        weights = [w / total_weight for w in weights]
        weights_global = weights

        # Calculate ESS
        ESS = 1.0 / sum([w**2 for w in weights])

        new_particlecloud = PoseArray()
        new_particlecloud.header.frame_id = "map"

        # Adjust particle count if ESS is very low or very high
        if ESS < 0.88 * self.particle_count:
            self.change_num_particles(2)
        elif ESS > 0.98 * self.particle_count:
            self.change_num_particles(-2)

        with open("weight_variance.txt", "a") as file:
            file.write(f"ESS: {ESS}, Particle Count: {self.particle_count}\n")

        # Inject random particles if ESS is below a threshold
        if ESS < 0.83 * self.particle_count:
           
         
                
            num_random = int(0.21 * self.particle_count)
            print("Injected random particles:", num_random)
            for _ in range(num_random):
                particle = Pose()
                particle.position.x = np.random.uniform(0, 3)
                particle.position.y = np.random.uniform(0, 3)
                particle.orientation = rotateQuaternion(Quaternion(), np.random.uniform(-math.pi, math.pi))
                new_particlecloud.poses.append(particle)

            indices = np.random.choice(len(self.particlecloud.poses), self.particle_count - num_random, p=weights)
            for index in indices:
                particle = self.particlecloud.poses[index]
                new_particle = Pose()
                new_particle.position.x = particle.position.x + np.random.normal(0, self.ODOM_TRANSLATION_NOISE)
                new_particle.position.y = particle.position.y + np.random.normal(0, self.ODOM_DRIFT_NOISE)
                new_particle.orientation = rotateQuaternion(particle.orientation, np.random.normal(0, self.ODOM_ROTATION_NOISE))
                new_particlecloud.poses.append(new_particle)
        else:
            # Normal resampling
            indices = np.random.choice(len(self.particlecloud.poses), int(self.particle_count * 0.95), p=weights)
            for index in indices:
                particle = self.particlecloud.poses[index]
                new_particle = Pose()
                new_particle.position.x = particle.position.x + np.random.normal(0, self.ODOM_TRANSLATION_NOISE)
                new_particle.position.y = particle.position.y + np.random.normal(0, self.ODOM_DRIFT_NOISE)
                new_particle.orientation = rotateQuaternion(particle.orientation, np.random.normal(0, self.ODOM_ROTATION_NOISE))
                new_particlecloud.poses.append(new_particle)

            # Adding motion-based noise to the lowest-weight particles
            num_random_particles = int(self.particle_count * 0.05)
            random_particles = np.random.choice(range(len(self.particlecloud.poses)), num_random_particles, replace=False)
            for index in random_particles:
                particle = self.particlecloud.poses[index]
                new_particle = Pose()
                new_particle.position.x = particle.position.x + np.random.normal(0, 0.4)  # Extra noise for exploration
                new_particle.position.y = particle.position.y + np.random.normal(0, 0.4)
                new_particle.orientation = rotateQuaternion(particle.orientation, np.random.normal(0, math.pi/4))
                new_particlecloud.poses.append(new_particle)

        # Update the particle cloud
        self.particlecloud = new_particlecloud


    def estimate_pose(self):
        

        #print(self.particle_count)
        particle_count = self.particle_count
        num_top_particles = int(0.60 * particle_count)

        # Get top-weighted particles' indices
        top_weight = np.argsort(weights_global)[-num_top_particles:]

        avg_x = 0
        avg_y = 0
        avg_q = Quaternion()

        # Sum up all positions and orientations
        for index in top_weight:
            if index < len(self.particlecloud.poses):
                particle = self.particlecloud.poses[index]  # Access particle by index
                avg_x += particle.position.x
                avg_y += particle.position.y
                avg_q.w += float(particle.orientation.w)
                avg_q.x += float(particle.orientation.x)
                avg_q.y += float(particle.orientation.y)
                avg_q.z += float(particle.orientation.z)
    
        # Get the average based on top-weighted particles
        avg_x /= num_top_particles
        avg_y /= num_top_particles
        avg_q.w /= num_top_particles
        avg_q.x /= num_top_particles
        avg_q.y /= num_top_particles
        avg_q.z /= num_top_particles
    
        # Normalize the quaternion to maintain valid orientation
        norm = math.sqrt(avg_q.w**2 + avg_q.x**2 + avg_q.y**2 + avg_q.z**2)
        avg_q.w /= norm
        avg_q.x /= norm
        avg_q.y /= norm
        avg_q.z /= norm
    
        # Create the Pose object
        estimated_pose = Pose()
        estimated_pose.position.x = avg_x
        estimated_pose.position.y = avg_y
        estimated_pose.orientation = avg_q
        
        return estimated_pose

    def change_num_particles(self, num_particles):
        """
        Set the number of particles used in the filter
        """
        #only change if its between max of 300 and min of 50
        if 50 <= self.particle_count + num_particles <= 250:
            self.particle_count += num_particles
        return self.particle_count