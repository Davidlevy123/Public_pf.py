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
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        
       
    def initialise_particle_cloud(self, initialpose):
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        
        particle_count = 100  # Number of particles
        self.particlecloud = PoseArray()
        self.particlecloud.header.frame_id = "map"
        
        for i in range(particle_count):
            # Add Gaussian noise to initial pose
            x_noise = np.random.normal(0, self.ODOM_TRANSLATION_NOISE)
            y_noise = np.random.normal(0, self.ODOM_DRIFT_NOISE)
            theta_noise = np.random.normal(0, self.ODOM_ROTATION_NOISE)


            pose = Pose()
            pose.position.x = initialpose.pose.pose.position.x + x_noise
            pose.position.y = initialpose.pose.pose.position.y + y_noise
            pose.orientation = rotateQuaternion(initialpose.pose.pose.orientation, theta_noise)

            self.particlecloud.poses.append(pose)

        
        
        return self.particlecloud

 
    
    def update_particle_cloud(self, scan):
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        weights = []
        
        # Calculate weights for each particle
        for particle in self.particlecloud.poses:
            weight = self.sensor_model.get_weight(scan, particle)
            weights.append(weight)
        
        # Normalise weights into probabilities 
        total_weight = sum(weights)
        #print("TOTAL SUM: " + str(total_weight))
        if total_weight == 0:
            return  # Skip if no valid weights

        # Normalize all weights correctly
        global weights_global
        weights = [w / total_weight for w in weights]
        weight_variance = np.var(weights)
        confidence_threshold_upper = 0.00002  # This threshold can be adjusted
        confidence_threshold_middle = 0.000002
        confidence_threshold_verylow = 0.0000005

        if weight_variance >= confidence_threshold_upper:
            confidence_status = "high"
        elif confidence_threshold_middle <= weight_variance < confidence_threshold_upper:
            confidence_status = "medium"
        elif confidence_threshold_middle >= weight_variance >= confidence_threshold_verylow:
            confidence_status = "low"
        else:
            confidence_status = "very low"
        
        print(f"\n\n\n\Weight Variance: {weight_variance}, Confidence Status: {confidence_status}")

        if(confidence_status == "low"):
            print("\n\n\n\Robot Lost")
        
        
        
        weights_global = weights

        

        # Resample particles based on weights using a probabilistic method (roulette-wheel)
        new_particlecloud = PoseArray()
        new_particlecloud.header.frame_id = "map"
        
        indices = np.random.choice(len(self.particlecloud.poses), len(self.particlecloud.poses), p=weights)
        for index in indices:
            new_particle = Pose()
            particle = self.particlecloud.poses[index]

            # Add noise during resampling
            new_particle.position.x = particle.position.x + np.random.normal(0, self.ODOM_TRANSLATION_NOISE)
            new_particle.position.y = particle.position.y + np.random.normal(0, self.ODOM_DRIFT_NOISE)
            new_particle.orientation = rotateQuaternion(particle.orientation, np.random.normal(0, self.ODOM_ROTATION_NOISE))
            
            new_particlecloud.poses.append(new_particle)
        
        # Attempting to sovle the kiddnapped robot problem by adding more noise to the bottom 5% of weighted particles 
        num_particles = len(new_particlecloud.poses)
        
        if confidence_status == "high":
            percentage_kiddnaped = 0.01
            noise_xy = 0.04 #standard noise values
            noise_rotation = 0.02
        elif confidence_status == "medium":
            percentage_kiddnaped = 0.05
            noise_xy = 0.4 # a bit higher
            noise_rotation = 0.05
        elif confidence_status == "low":
            percentage_kiddnaped = 0.15
            noise_xy = 2
            noise_rotation = 0.1
        elif confidence_status == "very low":
            percentage_kiddnaped = 0.5
            noise_xy = 10
            noise_rotation = 1
        kiddnapped_particles = int(percentage_kiddnaped * num_particles)

        low_weight_particles = np.argsort(weights)[:kiddnapped_particles]

        for index in low_weight_particles:
            low_weight_particle = new_particlecloud.poses[index]
            low_weight_particle.position.x += np.random.normal(0, noise_xy)  # Larger noise
            low_weight_particle.position.y += np.random.normal(0, noise_xy)
            low_weight_particle.orientation = rotateQuaternion(low_weight_particle.orientation, np.random.normal(0, noise_rotation))

        
        self.particlecloud = new_particlecloud

    def estimate_pose(self):
        


        particle_count = len(self.particlecloud.poses)
        num_top_particles = int(0.99 * particle_count)

        # Get top-weighted particles' indices
        top_weight = np.argsort(weights_global)[-num_top_particles:]

        avg_x = 0
        avg_y = 0
        avg_q = Quaternion()

        # Sum up all positions and orientations
        for index in top_weight:
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
