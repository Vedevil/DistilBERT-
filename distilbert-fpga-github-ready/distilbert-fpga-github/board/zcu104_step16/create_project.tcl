# Step16 integration scaffold, NOT executed in Vivado here.
# Usage: vivado -mode batch -source create_project.tcl -tclargs /NEW/project_dir
# Creates/validates BD only. NO launch_runs, bitstream or synthesis in this step.
set here [file dirname [file normalize [info script]]]
set root [file normalize [file join $here ../..]]
if {$argc != 1} {error "Supply a NEW project directory"}
set dest [file normalize [lindex $argv 0]]
if {[file exists $dest]} {error "Refusing existing project directory $dest"}
set boards [get_board_parts -quiet *zcu104*]
if {[llength $boards] == 0} {error "Install the ZCU104 board files matching your Vivado version first"}
set board [lindex [lsort -dictionary $boards] end]
create_project distilbert_zcu104 $dest -part xczu7ev-ffvc1156-2-e
set_property board_part $board [current_project]
add_files -norecurse [glob [file join $root rtl *.sv]]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1
create_bd_design system
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {64} CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] $ps
set accel [create_bd_cell -type module -reference distilbert_axi_q distilbert_0]
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl_connect]
set ddr [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ddr_connect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ddr
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset_0]
set_property CONFIG.C_EXT_RESET_HIGH {0} $reset
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* one]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] $one
connect_bd_net [get_bd_pins one/dout] [get_bd_pins reset_0/dcm_locked]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset_0/ext_reset_in]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins reset_0/slowest_sync_clk] [get_bd_pins ctrl_connect/aclk] [get_bd_pins ddr_connect/aclk] [get_bd_pins distilbert_0/clk] [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins reset_0/interconnect_aresetn] [get_bd_pins ctrl_connect/aresetn] [get_bd_pins ddr_connect/aresetn]
connect_bd_net [get_bd_pins reset_0/peripheral_aresetn] [get_bd_pins distilbert_0/aresetn]
foreach p {ps/M_AXI_HPM0_FPD ps/S_AXI_HP0_FPD distilbert_0/S_AXI distilbert_0/M_AXI} {
 if {[llength [get_bd_intf_pins -quiet $p]] != 1} {error "Expected interface $p missing; inspect Vivado interface inference/version before proceeding"}
}
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl_connect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl_connect/M00_AXI] [get_bd_intf_pins distilbert_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins distilbert_0/M_AXI] [get_bd_intf_pins ddr_connect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ddr_connect/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
assign_bd_address
# Do NOT hard-code a Python MMIO base: PYNQ reads the generated .hwh map.
validate_bd_design
save_bd_design
generate_target all [get_files system.bd]
set wrappers [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $wrappers
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "Step16 BD scaffold generated for $board. NO SYNTHESIS/IMPLEMENTATION/BITSTREAM EXECUTED."
puts "Before building: inspect generated address map, HP0 DDR visibility, clock/reset connectivity, Linux/PYNQ preset compatibility, and memory allocation limits."
