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
        self.particle_count = 100  # Number of particles
        self.confidence_threshold_upper = 0
        self.confidence_threshold_verylow = 1
       

        self.variance_window = []
        self.variance_window_size = 5
        self.same_confidence_status_count = 0
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

        # Normalize all weights correctly so that they represent percetages
        global weights_global
        weights = [w / total_weight for w in weights]
        
        weight_variance = np.var(weights)
        weights_global = weights


        normalised_variance = weight_variance / self.particle_count

        self.variance_window.append(normalised_variance)
        if len(self.variance_window) > self.variance_window_size:
            self.variance_window.pop(0)

        smooth_variance = sum(self.variance_window) / len(self.variance_window)




        #Writing the particle count and weight variance to a file
        #with open("weight_variance.txt", "a") as file:
            #write number of particles and weight variance with headings 
            #file.write(f"{self.particle_count} {weight_variance}\n")



        #scaling_factor = math.exp(-1170000*weight_variance)
        #print(f"Scaling Factor: {scaling_factor}")

        #set the confidence thresholds to max and min of the weight variance
        

        if smooth_variance >= self.confidence_threshold_upper:
            self.confidence_threshold_upper = smooth_variance
        if smooth_variance <= self.confidence_threshold_verylow:
            self.confidence_threshold_verylow = smooth_variance
        
        
       


        confidence_threshold_middle = ((self.confidence_threshold_upper + self.confidence_threshold_verylow) / 4)

        #making the upper confidence 90% of max and very low 5% more than min
        upper = 0.80 * self.confidence_threshold_upper
        lower = 1.2 * self.confidence_threshold_verylow
        print(f"Confidence Thresholds: {self.confidence_threshold_upper}, {confidence_threshold_middle}, {self.confidence_threshold_verylow}")
        
        if smooth_variance >= upper:
            confidence_status = "high"
        elif confidence_threshold_middle <= smooth_variance < upper:
            confidence_status = "medium"
        elif confidence_threshold_middle >= smooth_variance >= lower:
            confidence_status = "low"
        else:
            confidence_status = "very low"
        
        #print(f"\n\n\n\Weight Variance: {weight_variance}, Confidence Status: {confidence_status}")
        print(f"Confidence Status: {confidence_status}")
        if(confidence_status == "low"):
            print("\n\n\n\Robot Lost")
        
        
        
        

        

        # Resample particles based on weights using a probabilistic method (roulette-wheel)
        new_particlecloud = PoseArray()
        new_particlecloud.header.frame_id = "map"
        
 
        
        
        
        if confidence_status == "high":
            percentage_kiddnaped = 0.01
            #self.change_num_particles(-1)
            noise_xy = 0.02 #standard noise values
            
            noise_rotation = 0.02
        elif confidence_status == "medium":
            percentage_kiddnaped = 0.05
            #self.change_num_particles(-1)
            noise_xy = 0.08 # a bit higher
            noise_rotation = 0.05
        elif confidence_status == "low":
            percentage_kiddnaped = 0.1
            #self.change_num_particles(1)
            noise_xy = 0.2
            noise_rotation = 0.1
        elif confidence_status == "very low":
            percentage_kiddnaped = 0.2
            #self.change_num_particles(1)
            noise_xy = 1
            noise_rotation = 0.5


        prev_confidence_status = confidence_status
        #only add or remove particles if the confidence status has been the same for 10 iterations
        if confidence_status == prev_confidence_status:
            self.same_confidence_status_count += 1
        else:
            self.same_confidence_status_count = 0
        if self.same_confidence_status_count >= 5:
            if confidence_status == "high":
                self.change_num_particles(-2)
            elif confidence_status == "medium":
                self.change_num_particles(-1)
            elif confidence_status == "low":
                self.change_num_particles(1)
            elif confidence_status == "very low":
                self.change_num_particles(2)
            self.same_confidence_status_count = 0
        


        indices = np.random.choice(len(self.particlecloud.poses), self.particle_count, p=weights)
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
        #noise_xy = 10*scaling_factor
        #noise_rotation = 2*scaling_factor
        print(f"num_particles: {num_particles}")
        #print(f"Noise XY: {noise_xy}")
        #percentage_kiddnaped = 0.8*scaling_factor
        kiddnapped_particles = int(percentage_kiddnaped * self.particle_count)
        low_weight_particles = np.argsort(weights)[:kiddnapped_particles]

        #compare len of new particlecloud.poses to the particle count in same print statement
        print(f"len of new particlecloud.poses: {len(new_particlecloud.poses)}, particle count: {self.particle_count}")
        

        for index in low_weight_particles:
            if index < len(new_particlecloud.poses):
                low_weight_particle = new_particlecloud.poses[index]
                low_weight_particle.position.x += np.random.normal(0, noise_xy)  # Larger noise
                low_weight_particle.position.y += np.random.normal(0, noise_xy)
                low_weight_particle.orientation = rotateQuaternion(low_weight_particle.orientation, np.random.normal(0, noise_rotation))


        
        self.particlecloud = new_particlecloud

    def estimate_pose(self):
        

        print(self.particle_count)
        particle_count = self.particle_count
        num_top_particles = int(0.99 * particle_count)

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
        if 50 <= self.particle_count + num_particles <= 300:
            self.particle_count += num_particles
        return self.particle_count
