use vulkanalia::prelude::v1_0::*;

use super::Allocate;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::{anyhow,Result};

/// Abstraction over memory used by Vulkan resources.
pub type Memory = vk::DeviceMemory;

pub struct Allocator {
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Allocator {
    pub fn new(memory_properties: vk::PhysicalDeviceMemoryProperties) -> Self {
        Self { memory_properties }
    }

    fn find_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for i in 0..self.memory_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return Ok(i);
            }
        }
        Err(anyhow!(
            "Failed to find suitable memory type."),
        )
    }
}

impl Allocate for Allocator {
    type Memory = Memory;

    fn create_buffer(
        &mut self,
        device: &Device,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Self::Memory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size as _)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.find_memory_type(
            mem_requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    }

    fn create_image(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> Result<(vk::Image, Self::Memory)> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::_1)
            .flags(vk::ImageCreateFlags::empty());

        let image = unsafe { device.create_image(&image_info, None)? };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index =
            self.find_memory_type(mem_requirements, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = unsafe {
            let mem = device.allocate_memory(&alloc_info, None)?;
            device.bind_image_memory(image, mem, 0)?;
            mem
        };

        Ok((image, memory))
    }

    fn destroy_buffer(
        &mut self,
        device: &Device,
        buffer: vk::Buffer,
        memory: Self::Memory,
    ) -> Result<()> {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }

        Ok(())
    }

    fn destroy_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        memory: Self::Memory,
    ) -> Result<()> {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }

        Ok(())
    }

    fn update_buffer<T: Copy>(
        &mut self,
        device: &Device,
        memory: &Self::Memory,
        data: &[T],
    ) -> Result<()> {
        let size = (data.len() * std::mem::size_of::<T>()) as _;
        unsafe {
            let data_ptr = device.map_memory(*memory, 0, size, vk::MemoryMapFlags::empty())?;
            memcpy(&data, data_ptr.cast(), size as usize);
            device.unmap_memory(*memory);
        }

        Ok(())
    }
}
